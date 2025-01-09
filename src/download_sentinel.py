# Standard Library Imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
from pathlib import Path
import os
import random

from tqdm.notebook import tqdm
import time

# Third-Party Imports
import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from lightgbm import LGBMClassifier
from shapely.affinity import scale, translate
from skimage import exposure
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from shapely import wkt
import geopandas as gpd

import warnings
warnings.filterwarnings('ignore')

import logging
# Set up a logger to capture Rasterio warnings
logging.getLogger("rasterio._env").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

import logging
# Set up a logger to capture Rasterio warnings
logging.getLogger("rasterio._env").setLevel(logging.ERROR)


# Define the root path for the project
root_path = Path("..")

# Initialize Earth Engine with a specific project
# Replace "project" with your project ID as needed
#ee.Authenticate()
ee.Initialize(project="ee-crop-health-telangana")

# Load training and testing datasets from CSV files
train = pd.read_csv(root_path / 'data/train.csv')
test = pd.read_csv(root_path / 'data/test.csv')

# Convert WKT geometry to actual geometry objects in both datasets
train['geometry'] = train['geometry'].apply(wkt.loads)
test['geometry'] = test['geometry'].apply(wkt.loads)

# Convert pandas DataFrames to GeoDataFrames with CRS set to 'epsg:4326'
train = gpd.GeoDataFrame(train, crs='epsg:4326')
test = gpd.GeoDataFrame(test, crs='epsg:4326')

# Concatenate train and test datasets into a single DataFrame for consistent processing
# 'dataset' column distinguishes between train and test rows
data = pd.concat(
    [train.assign(dataset='train'), test.assign(dataset='test')]
).reset_index(drop=True)



def setup_download_environment(base_dir='../data/sentinel-2-all'):
    os.makedirs(base_dir, exist_ok=True)
    ee.Initialize()
    return base_dir

def process_single_farm(row, output_dir, cloud_threshold=20):
    try:
        harvest_date = pd.to_datetime(row['HDate'])
        temporal_files = []
        
        # Time points: -10, -5, 0 (harvest), +5 days
        time_points = [-10, -5, 0, 5]
        
        for days_offset in time_points:
            target_date = harvest_date + timedelta(days=days_offset)
            file_name = f"S2_{row['FarmID']}_{target_date.strftime('%Y%m%d')}.tif"
            output_path = os.path.join(output_dir, file_name)
            
            if os.path.exists(output_path):
                temporal_files.append(output_path)
                continue
                
            # Narrow window to find closest image (2 days before and after target)
            start_date = (target_date - timedelta(days=2)).strftime('%Y-%m-%d')
            end_date = (target_date + timedelta(days=2)).strftime('%Y-%m-%d')
            
            region = ee.Geometry.Polygon(row['geometry'].__geo_interface__['coordinates']) if row['geometry'].geom_type == 'Polygon' else \
                     ee.Geometry.MultiPolygon([polygon.exterior.coords[:] for polygon in row['geometry'].geoms])
            
            collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
                .filterDate(start_date, end_date) \
                .filterBounds(region) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)) \
                .select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])
            
            if collection.size().getInfo() == 0:
                continue
                
            # Get image closest to target date
            def add_date_distance(image):
                image_date = ee.Date(image.get('system:time_start'))
                target_ee_date = ee.Date(target_date.strftime('%Y-%m-%d'))
                diff = ee.Number(image_date.difference(target_ee_date, 'day')).abs()
                return image.set('date_diff', diff)
                
            closest_image = collection.map(add_date_distance) \
                .sort('date_diff') \
                .first() \
                .clip(region)
            
            for attempt in range(3):
                try:
                    geemap.ee_export_image(
                        closest_image,
                        filename=output_path,
                        scale=10,
                        region=region,
                        file_per_band=False,
                        crs='EPSG:4326'
                    )
                    time.sleep(1)
                    temporal_files.append(output_path)
                    break
                except Exception as e:
                    time.sleep(2 ** attempt)
        
        return temporal_files if temporal_files else None
        
    except Exception as e:
        print(f"Error processing FarmID {row['FarmID']}: {str(e)}")
        return None

def download_and_update_dataset(df, output_dir, max_workers=4):
    df = df.copy()
    df['tif_paths'] = None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_farm, row, output_dir): idx 
            for idx, row in df.iterrows()
        }
        
        for future in tqdm(as_completed(futures), total=len(df)):
            idx = futures[future]
            try:
                result = future.result()
                df.at[idx, 'tif_paths'] = result
            except Exception as e:
                print(f"Failed to process index {idx}: {str(e)}")
    
    return df
# Usage

output_dir = setup_download_environment()
enriched_data = download_and_update_dataset(data.head(5), output_dir)