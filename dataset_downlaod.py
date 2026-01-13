# download_data.py
"""
Simple STURM-Flood dataset download & extract
Uses: /workspace/geoai/flood_data/sturm/Dataset/
"""

import os
import zipfile
import subprocess
from pathlib import Path
from tqdm import tqdm
import requests

# Your exact path
BASE_DIR = Path('/workspace/geoai/flood_data')
STURM_DIR = BASE_DIR / 'sturm'

# URL
URL = "https://zenodo.org/records/12748983/files/Dataset.zip?download=1"
ZIP_FILE = BASE_DIR / "sturm_dataset.zip"

def download_file(url, filepath):
    """Download with progress bar"""
    print(f"📥 Downloading to {filepath}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"📂 Extracting to {extract_to}...")
    
    try:
        subprocess.run(['unzip', '-q', str(zip_path), '-d', str(extract_to)], 
                      timeout=300, check=True)
    except:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

if __name__ == "__main__":
    print("="*70)
    print("STURM-FLOOD DATA DOWNLOADER")
    print("="*70)
    
    # Create directory
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Base directory: {BASE_DIR}")
    
    # Download
    if not ZIP_FILE.exists():
        download_file(URL, ZIP_FILE)
        print(f"✓ Downloaded: {ZIP_FILE}")
    else:
        print(f"✓ Already exists: {ZIP_FILE}")
    
    # Extract to BASE_DIR (so it creates sturm/Dataset/ inside)
    extract_zip(ZIP_FILE, BASE_DIR)
    print(f"✓ Extracted")
    
    # Cleanup
    os.remove(ZIP_FILE)
    print(f"✓ Cleanup: Removed zip file")
    
    # Show final structure
    print(f"\n📂 Your data structure:")
    print(f"   {BASE_DIR}/")
    print(f"   └── sturm/")
    print(f"       └── Dataset/")
    print(f"           ├── Sentinel1/")
    print(f"           │   ├── S1/")
    print(f"           │   └── Floodmaps/")
    print(f"           └── Sentinel2/")
    print(f"               ├── S2/")
    print(f"               └── Floodmaps/")
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print(f"✓ Path: {STURM_DIR}")
    print("="*70)
