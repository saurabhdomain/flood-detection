import os
import zipfile
import subprocess
from pathlib import Path
from tqdm import tqdm
import requests
import shutil

# Your exact path
BASE_DIR = Path('/workspace/geoai/flood_data')
STURM_DIR = BASE_DIR / 'sturm'

# URL
URL = "https://zenodo.org/records/12748983/files/Dataset.zip?download=1"
ZIP_FILE = BASE_DIR / "sturm_dataset.zip"

def download_file(url, filepath):
    """Download with progress bar"""
    print(f"📥 Downloading to {filepath}...")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # ← Add error checking
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # ← Filter out keep-alive chunks
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✓ Download complete: {filepath.stat().st_size / 1e9:.2f} GB")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        if filepath.exists():
            filepath.unlink()  # Remove partial download
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"📂 Extracting to {extract_to}...")
    
    try:
        # Try fast unzip command first
        result = subprocess.run(
            ['unzip', '-q', str(zip_path), '-d', str(extract_to)], 
            timeout=600,  # ← Increase timeout to 10 minutes
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ Extracted using system unzip")
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"⚠️  System unzip failed ({e}), trying Python zipfile...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                with tqdm(total=len(members), desc="Extracting") as pbar:
                    for member in members:
                        zip_ref.extract(member, extract_to)
                        pbar.update(1)
            
            print("✓ Extracted using Python zipfile")
            return True
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False

def verify_structure(base_dir):
    """Verify the extracted data structure"""
    print("\n🔍 Verifying data structure...")
    
    expected_paths = [
        base_dir  / 'Dataset' / 'Sentinel1' / 'S1',
        base_dir  / 'Dataset' / 'Sentinel1' / 'Floodmaps',
        base_dir  / 'Dataset' / 'Sentinel2' / 'S2',
        base_dir  / 'Dataset' / 'Sentinel2' / 'Floodmaps',
    ]
    
    all_exist = True
    for path in expected_paths:
        if path.exists():
            file_count = len(list(path.glob('*.tif'))) + len(list(path.glob('*.tiff')))
            print(f"  ✓ {path.relative_to(base_dir)}: {file_count} files")
        else:
            print(f"  ❌ Missing: {path.relative_to(base_dir)}")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print("="*70)
    print("STURM-FLOOD DATA DOWNLOADER")
    print("="*70)
    
    # Ask for confirmation if data already exists
    if STURM_DIR.exists():
        print(f"\n⚠️  Data directory already exists: {STURM_DIR}")
        response = input("Do you want to delete and re-download? (yes/no): ")
        if response.lower() == 'yes':
            print(f"🗑️  Removing existing data...")
            shutil.rmtree(STURM_DIR)
            print(f"✓ Removed: {STURM_DIR}")
        else:
            print("❌ Cancelled. Keeping existing data.")
            exit(0)
    
    # Create directory
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Base directory: {BASE_DIR}")
    
    # Download
    if not ZIP_FILE.exists():
        success = download_file(URL, ZIP_FILE)
        if not success:
            print("❌ Download failed. Exiting.")
            exit(1)
    else:
        print(f"✓ Zip file already exists: {ZIP_FILE}")
        print(f"  Size: {ZIP_FILE.stat().st_size / 1e9:.2f} GB")
    
    # Extract
    success = extract_zip(ZIP_FILE, BASE_DIR)
    if not success:
        print("❌ Extraction failed. Exiting.")
        exit(1)
    
    # Verify structure
    if verify_structure(BASE_DIR):
        print("\n✓ Data structure verified!")
    else:
        print("\n⚠️  Data structure incomplete - check extraction")
    
    # Cleanup
    try:
        os.remove(ZIP_FILE)
        print(f"\n✓ Cleanup: Removed zip file")
    except Exception as e:
        print(f"\n⚠️  Could not remove zip: {e}")
    
    # Show final structure
    print(f"\n📂 Your data structure:")
    print(f"   {BASE_DIR}/")
    print(f"       └── Dataset/")
    print(f"           ├── Sentinel1/")
    print(f"           │   ├── S1/")
    print(f"           │   └── Floodmaps/")
    print(f"           └── Sentinel2/")
    print(f"               ├── S2/")
    print(f"               └── Floodmaps/")
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print(f"✓ Dataset path: {STURM_DIR}")
    print(f"✓ Config should use: data_dir: '/workspace/geoai/flood_data/sturm'")
    print("="*70)