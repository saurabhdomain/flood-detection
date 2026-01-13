"""
00_setup.py - Verify all dependencies are installed correctly
Run this ONCE at the beginning
"""

import sys
import torch
import numpy as np

print("\n" + "="*60)
print("SETUP VERIFICATION")
print("="*60 + "\n")

# Check Python
print(f"✓ Python {sys.version.split()[0]}")

# Check PyTorch
print(f"✓ PyTorch {torch.__version__}")

# Check GPU
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ GPU NOT available (will use CPU)")

# Test imports
try:
    import segmentation_models_pytorch as smp
    print("✓ segmentation-models-pytorch installed")
except:
    print("✗ segmentation-models-pytorch NOT installed")
    sys.exit(1)

try:
    import rasterio
    print("✓ rasterio installed")
except:
    print("✗ rasterio NOT installed")
    sys.exit(1)

try:
    import cv2
    print("✓ opencv-python installed")
except:
    print("✗ opencv-python NOT installed")
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL SETUP CHECKS PASSED!")
print("="*60 + "\n")

# Test GPU with dummy tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(2, 3, 256, 256).to(device)
print(f"✓ Test tensor shape: {x.shape}")
print(f"✓ Test tensor device: {x.device}")
print("\nYou're ready to start!\n")
