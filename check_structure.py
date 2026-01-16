# Replace the verify_structure function with this smart version:
from pathlib import Path
def find_data_directories(base_dir):
    """Intelligently find where the actual data is"""
    print("\n🔍 Searching for data directories...")
    
    # Look for S1, S2, and Floodmaps directories
    s1_dirs = list(base_dir.rglob('S1'))
    s2_dirs = list(base_dir.rglob('S2'))
    floodmap_dirs = list(base_dir.rglob('Floodmaps'))
    
    print(f"  Found {len(s1_dirs)} 'S1' directories")
    print(f"  Found {len(s2_dirs)} 'S2' directories")
    print(f"  Found {len(floodmap_dirs)} 'Floodmaps' directories")
    
    # Find the ones with TIF files
    valid_s1 = None
    valid_s2 = None
    valid_s1_floods = None
    valid_s2_floods = None
    
    for s1_dir in s1_dirs:
        tif_count = len(list(s1_dir.glob('*.tif'))) + len(list(s1_dir.glob('*.tiff')))
        if tif_count > 0:
            valid_s1 = s1_dir
            print(f"  ✓ Found S1 data: {s1_dir.relative_to(base_dir)} ({tif_count} files)")
            
            # Look for corresponding Floodmaps
            parent = s1_dir.parent
            potential_flood = parent / 'Floodmaps'
            if potential_flood.exists():
                flood_count = len(list(potential_flood.glob('*.tif'))) + len(list(potential_flood.glob('*.tiff')))
                if flood_count > 0:
                    valid_s1_floods = potential_flood
                    print(f"  ✓ Found S1 Floodmaps: {potential_flood.relative_to(base_dir)} ({flood_count} files)")
    
    for s2_dir in s2_dirs:
        tif_count = len(list(s2_dir.glob('*.tif'))) + len(list(s2_dir.glob('*.tiff')))
        if tif_count > 0:
            valid_s2 = s2_dir
            print(f"  ✓ Found S2 data: {s2_dir.relative_to(base_dir)} ({tif_count} files)")
            
            # Look for corresponding Floodmaps
            parent = s2_dir.parent
            potential_flood = parent / 'Floodmaps'
            if potential_flood.exists():
                flood_count = len(list(potential_flood.glob('*.tif'))) + len(list(potential_flood.glob('*.tiff')))
                if flood_count > 0:
                    valid_s2_floods = potential_flood
                    print(f"  ✓ Found S2 Floodmaps: {potential_flood.relative_to(base_dir)} ({flood_count} files)")
    
    if valid_s1 and valid_s2 and valid_s1_floods and valid_s2_floods:
        print("\n✓ All required directories found!")
        
        # Show the correct path to use in config
        data_root = valid_s1.parent.parent  # Go up to Dataset level
        print(f"\n📝 Use this in your config.yaml:")
        print(f"   data_dir: '{data_root}'")
        
        return True, data_root
    else:
        print("\n❌ Missing some required directories")
        return False, None

# Update the main block:

if __name__ == "__main__":
    print("="*70)
    print("STURM-FLOOD DATA DOWNLOADER")
    print("="*70)
    
    # ... existing download code ...
    BASE_DIR = Path('/workspace/geoai/flood_data')
    # Verify structure (use new smart function)
    success, data_root = find_data_directories(BASE_DIR)
    
    if success:
        print("\n✓ Data structure verified!")
        print(f"\n📂 Your actual data structure:")
        print(f"   {data_root}/")
        print(f"   ├── Sentinel1/")
        print(f"   │   ├── S1/ (Sentinel-1 SAR images)")
        print(f"   │   └── Floodmaps/ (S1 flood masks)")
        print(f"   └── Sentinel2/")
        print(f"       ├── S2/ (Sentinel-2 optical images)")
        print(f"       └── Floodmaps/ (S2 flood masks)")
        
        print("\n" + "="*70)
        print("✓ COMPLETE!")
        print(f"✓ Dataset path: {data_root}")
        print(f"\n📝 Update your config.yaml:")
        print(f"   data:")
        print(f"     data_dir: '{data_root}'")
        print("="*70)
    else:
        print("\n⚠️  Data structure incomplete or in unexpected format")
        print("\n💡 Run this to debug:")
        print("   python check_structure.py")