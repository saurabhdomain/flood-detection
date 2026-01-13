"""
03_model.py - Create model with correct channels
Auto-calculates from config modality
Runs on GPU/CPU based on config
"""

import torch
import segmentation_models_pytorch as smp
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Get device from config
DEVICE = torch.device(CONFIG['training']['device'])

print(f"\n🖥️  Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

## Read Activation function from config
activattion_function = CONFIG['model']['activation']

def get_input_channels(modality):
    """Get input channels from modality string"""
    if modality == "s1":
        return 2  # VV, VH
    elif modality == "s2":
        return 11  # Optical bands
    elif modality == "s1_s2":
        return 13  # 2 + 11
    else:
        raise ValueError(f"Unknown modality: {modality}")

def create_model(modality, encoder_name='resnet18', device=DEVICE):
    """Create U-Net with correct channels and move to device"""
    
    in_channels = get_input_channels(modality)
    
    print(f"\nCreating model:")
    print(f"  Modality: {modality}")
    print(f"  Input channels: {in_channels}")
    print(f"  Encoder: {encoder_name}")
    
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights='imagenet',
        in_channels=in_channels,
        classes=1,
        activation=None
    )
    
    # IMPORTANT: Move model to device (GPU or CPU)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Device: {device}\n")
    
    return model

# Test
if __name__ == "__main__":
    model = create_model(
        CONFIG['data']['modality'],
        CONFIG['model']['encoder_name'],
        device=DEVICE
    )
    
    # Test with dummy input ON SAME DEVICE
    in_ch = get_input_channels(CONFIG['data']['modality'])
    dummy = torch.randn(1, in_ch, 256, 256).to(DEVICE)  # ← Important: .to(DEVICE)
    out = model(dummy)
    print(f"  Input shape: {dummy.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Output device: {out.device}")
    print(f"✓ Model works on {DEVICE}!")
