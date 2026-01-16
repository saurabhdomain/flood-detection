"""
01_explore_data.py - Explore your downloaded dataset
This helps you understand your data structure BEFORE coding
"""

import os
from pathlib import Path
import rasterio
import numpy as np
from tqdm import tqdm

# MODIFY THIS PATH to your data
DATA_PATH = "/workspace/geoai/flood_data/"  # ← CHANGE THIS

def explore_directory(path, max_depth=3, current_depth=0):
    """Print directory structure nicely"""
    path = Path(path)
    
    if current_depth > max_depth:
        return
    
    indent = "  " * current_depth
    
    try:
        items = sorted(os.listdir(path))
    except:
        return
    
    for item in items[:20]:  # Show first 20 items
        item_path = path / item
        if os.path.isdir(item_path):
            print(f"{indent}📁 {item}/")
            explore_directory(item_path, max_depth, current_depth + 1)
        else:
            size_mb = os.path.getsize(item_path) / (1024*1024)
            print(f"{indent}📄 {item} ({size_mb:.1f} MB)")

def analyze_tif_file(filepath):
    """Show what's inside a .tif file"""
    try:
        with rasterio.open(filepath) as src:
            print(f"\n  Shape: {src.shape} (height, width)")
            print(f"  Channels: {src.count}")
            print(f"  Dtype: {src.dtypes[0]}")
            
            # Read and show stats
            data = src.read()
            print(f"  Min: {data.min():.2f}, Max: {data.max():.2f}, Mean: {data.mean():.2f}")
    except Exception as e:
        print(f"  Error reading: {e}")

# Main exploration
print("\n" + "="*70)
print("EXPLORING YOUR DATASET")
print("="*70 + "\n")

data_path = Path(DATA_PATH)

if not data_path.exists():
    print(f"✗ Path not found: {DATA_PATH}")
    print("Please modify DATA_PATH in this script!")
    exit(1)

print(f"Exploring: {data_path}\n")

# Show structure
explore_directory(data_path, max_depth=3)

# Find all .tif files
print("\n" + "="*70)
print("FINDING .TIF FILES")
print("="*70 + "\n")

tif_files = list(data_path.glob("**/*.tif"))
print(f"Total .tif files: {len(tif_files)}\n")

if tif_files:
    print("Sample .tif files:")
    for tif_file in tif_files[:5]:
        print(f"\n📍 {tif_file.relative_to(data_path)}")
        analyze_tif_file(tif_file)

# Identify data organization
print("\n" + "="*70)
print("YOUR DATA ORGANIZATION")
print("="*70 + "\n")

s1_tiles = list(data_path.glob("**/Sentinel1/**/S1/*.tif"))
s1_masks = list(data_path.glob("**/Sentinel1/**/Floodmaps/*.tif"))
s2_tiles = list(data_path.glob("**/Sentinel2/**/S2/*.tif"))

print(f"Sentinel-1 SAR tiles: {len(s1_tiles)}")
print(f"Sentinel-1 Masks: {len(s1_masks)}")
print(f"Sentinel-2 Tiles: {len(s2_tiles)}")

print("\n✓ Exploration complete!")
print("\nNEXT STEP: Run 02_dataset.py to load this data\n")
