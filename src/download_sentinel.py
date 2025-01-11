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



import ee
import geemap
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from tqdm.notebook import tqdm
import time
import json

import ee
import geemap
import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from tqdm.notebook import tqdm
import time
import json

import ee
import geemap
import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from tqdm.notebook import tqdm
import time
import json
import geopandas as gpd

def setup_download_environment(base_dir='../data/sentinel-2-all'):
    """Setup directory structure including progress tracking"""
    os.makedirs(base_dir, exist_ok=True)
    progress_dir = os.path.join(base_dir, 'progress')
    os.makedirs(progress_dir, exist_ok=True)
    ee.Initialize()
    return base_dir, progress_dir

def load_progress(progress_dir):
    """Load progress from checkpoint file"""
    progress_file = os.path.join(progress_dir, 'download_progress.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'completed_farms': [], 'last_index': 0}

def save_progress(progress_dir, completed_farms, last_index):
    """Save progress to checkpoint file with type conversion"""
    progress_file = os.path.join(progress_dir, 'download_progress.json')
    progress_data = {
        'completed_farms': [int(x) if isinstance(x, np.integer) else x for x in completed_farms],
        'last_index': int(last_index)
    }
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f)

def process_single_farm(row, output_dir, cloud_threshold=20):
    """Process single farm with flexible date selection"""
    try:
        harvest_date = pd.to_datetime(row['HDate'])
        temporal_files = []
        farm_id = row['FarmID']
        
        print(f"\nProcessing FarmID: {farm_id}")
        
        # Time points and their windows (target offset, search window in days)
        time_points = [
            (-10, 5),  # For -15 days, search ±10 days
            (-5, 5),  # For -10 days, search ±10 days
            (0, 5),    # For harvest date, search ±10 days
            (5, 5)    # For +10 days, search ±10 days
        ]
        
        for target_offset, window_size in time_points:
            target_date = harvest_date + timedelta(days=target_offset)
            file_name = f"S2_{farm_id}_{target_date.strftime('%Y%m%d')}.tif"
            output_path = os.path.join(output_dir, file_name)
            
            if os.path.exists(output_path):
                print(f"File exists for {farm_id} at offset {target_offset}")
                temporal_files.append(output_path)
                continue
            
            print(f"Searching images for {farm_id} around offset {target_offset}")
            
            # Wider search window
            start_date = (target_date - timedelta(days=window_size)).strftime('%Y-%m-%d')
            end_date = (target_date + timedelta(days=window_size)).strftime('%Y-%m-%d')
            
            region = ee.Geometry.Polygon(row['geometry'].__geo_interface__['coordinates']) if row['geometry'].geom_type == 'Polygon' else \
                     ee.Geometry.MultiPolygon([polygon.exterior.coords[:] for polygon in row['geometry'].geoms])
            
            collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
                .filterDate(start_date, end_date) \
                .filterBounds(region) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)) \
                .select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])
            
            collection_size = collection.size().getInfo()
            if collection_size == 0:
                print(f"No images found for {farm_id} around offset {target_offset} within ±{window_size} days")
                continue
            
            print(f"Found {collection_size} images for {farm_id} around offset {target_offset}")
                
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
                    print(f"Successfully downloaded {farm_id} at offset {target_offset}")
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for {farm_id} at offset {target_offset}: {str(e)}")
                    time.sleep(2 ** attempt)
        
        return temporal_files if temporal_files else None
        
    except Exception as e:
        print(f"Error processing FarmID {farm_id}: {str(e)}")
        return None

def download_and_update_dataset(original_gdf, output_dir, progress_dir, batch_size=100, max_workers=4):
    """Download images with progress tracking and periodic saves"""
    # Load existing progress
    progress = load_progress(progress_dir)
    completed_farms = progress['completed_farms']
    start_idx = progress['last_index']
    
    print(f"Resuming from index {start_idx}")
    print(f"Found {len(completed_farms)} completed farms")
    
    # Load existing results if available
    results_path = os.path.join(output_dir, 'sentinel_downloads.csv')
    if os.path.exists(results_path):
        print("Loading existing results file")
        results_df = pd.read_csv(results_path)
        gdf = original_gdf.merge(results_df[['FarmID', 'tif_paths']], 
                               on='FarmID', 
                               how='left')
    else:
        print("Starting fresh download")
        gdf = original_gdf.copy()
        gdf['tif_paths'] = None
    
    # Reset progress if needed
    if start_idx >= len(gdf):
        print("Resetting progress as start_index exceeds dataset length")
        start_idx = 0
        completed_farms = []
    
    total_farms = len(gdf)
    remaining_farms = total_farms - len(completed_farms)
    print(f"Total farms: {total_farms}")
    print(f"Remaining farms: {remaining_farms}")
    
    # Process in batches
    for batch_start in range(start_idx, len(gdf), batch_size):
        batch_end = min(batch_start + batch_size, len(gdf))
        batch_df = gdf.iloc[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start} to {batch_end}")
        
        # Filter out completed farms
        batch_to_process = batch_df[~batch_df['FarmID'].astype(str).isin([str(x) for x in completed_farms])]
        print(f"Farms to process in this batch: {len(batch_to_process)}")
        
        if len(batch_to_process) == 0:
            print("No farms to process in this batch, moving to next")
            continue
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_farm, row, output_dir): idx 
                for idx, row in batch_to_process.iterrows()
            }
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx = futures[future]
                try:
                    result = future.result()
                    gdf.at[idx, 'tif_paths'] = str(result) if result else None
                    if result:
                        completed_farms.append(int(gdf.at[idx, 'FarmID']))
                except Exception as e:
                    print(f"Failed to process index {idx}: {str(e)}")
        
        # Save progress after each batch
        save_progress(progress_dir, completed_farms, batch_end)
        
        # Save results without geometry column to CSV
        results_df = pd.DataFrame({
            'FarmID': gdf['FarmID'],
            'tif_paths': gdf['tif_paths']
        })
        results_df.to_csv(results_path, index=False)
        
        print(f"Progress saved. Completed farms: {len(completed_farms)}")
        time.sleep(5)
    
    return gdf

# Usage
# Reset progress if needed (uncomment if you want to start fresh)
# if os.path.exists('../data/sentinel-2-all/progress/download_progress.json'):
#     os.remove('../data/sentinel-2-all/progress/download_progress.json')

# Usage example:
# Assuming 'data' is your original GeoDataFrame with geometry and FarmID
output_dir, progress_dir = setup_download_environment()
enriched_data = download_and_update_dataset(data, output_dir, progress_dir)

# Save the DataFrame to CSV
csv_path = os.path.join(output_dir, 'sentinel_downloads.csv')
enriched_data.to_csv(csv_path, index=False)
print(f"Results saved to: {csv_path}")





