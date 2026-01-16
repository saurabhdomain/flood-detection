"""
Validate dataset for corrupted or problematic tiles
"""
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("/workspace/geoai/flood_data/Dataset")
S1_DIR = DATA_DIR / "Sentinel1" / "S1"
MASK_DIR = DATA_DIR / "Sentinel1" / "Floodmaps"

def check_file(filepath):
    """Check a single file for issues"""
    issues = []
    try:
        with rasterio.open(filepath) as src:
            data = src.read().astype(np.float32)
            
            if np.isnan(data).any():
                issues.append("contains NaN")
            if np.isinf(data).any():
                issues.append("contains Inf")
            if data.std() == 0:
                issues.append("zero variance (constant)")
            if data.max() == data.min():
                issues.append("all same value")
            if np.abs(data).max() > 1e10:
                issues.append("extreme values")
                
    except Exception as e:
        issues.append(f"read error: {e}")
    
    return issues

print("Checking S1 tiles...")
problematic_files = []

for f in tqdm(list(S1_DIR.glob("*.tif"))):
    issues = check_file(f)
    if issues:
        problematic_files.append((f.name, issues))
        print(f"⚠️  {f.name}: {issues}")

print(f"\n{'='*60}")
print(f"Total problematic files: {len(problematic_files)}")

if problematic_files:
    print("\nProblematic files:")
    for name, issues in problematic_files[:20]:
        print(f"  - {name}: {issues}")