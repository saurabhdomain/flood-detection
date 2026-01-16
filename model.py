"""
03_model.py - Create model with correct channels
Auto-calculates from config modality
Runs on GPU/CPU based on config
"""

import torch
import segmentation_models_pytorch as smp
import yaml
import torch.nn as nn
import os
from pathlib import Path



script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Get device from config
DEVICE = torch.device(CONFIG['training']['device'])

print(f"\n🖥️  Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

## Read Activation function from config
activattion_function = CONFIG['model']['activation']

class UNetWithDropout(nn.Module):

    """
    U-net wrapper with dropout to prevent overfitting by deactivation of neuron during training. Its a way of regularization.
    """
    def __init__(self, base_model, dropout_rate =0.3):
         """
        Args:
            base_model: The base U-Net model from segmentation_models_pytorch
            dropout_rate: Probability of dropping a neuron (typical: 0.2-0.5)
        """
         super().__init__()
         self.model = base_model
         self.dropout = nn.Dropout2d(p=dropout_rate) # scaling handling
         self.dropout_rate = dropout_rate

    def forward(self, x):
     
        """
        Forward pass with dropout applied
        """
        # Get encoder features
        features = self.model.encoder(x)
        
        # Decoder with dropout
        decoder_output = self.model.decoder(features)
        
        # Apply dropout after decoder
        # nn.Dropout2d automatically:
        # 1. If training: Generate mask, apply it, scale by 1/(1-p)
        # 2. If eval: Do nothing (pass through unchanged)
        
        if self.training:
            decoder_output = self.dropout(decoder_output)
        # Segmentation head
        output = self.model.segmentation_head(decoder_output)
        
        return output
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

def create_model(modality, encoder_name='resnet18', architecture='unet', device=DEVICE, dropout_rate=0.3):
    """
    Create segmentation model with correct channels and move to device
    
    Args:
        modality: Data modality (s1, s2, s1_s2)
        encoder_name: Encoder backbone (resnet18, resnet34, efficientnet-b0, etc.)
        architecture: Model architecture (unet, deeplabv3+, fpn, pspnet)
        device: Device to move model to
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Model ready for training
    """
    in_channels = get_input_channels(modality)
    
    print(f"\nCreating model:")
    print(f"  Architecture: {architecture}")
    print(f"  Modality: {modality}")
    print(f"  Input channels: {in_channels}")
    print(f"  Encoder: {encoder_name}")
    print(f"  Dropout rate: {dropout_rate}")
    
    # Select architecture
    if architecture.lower() == 'unet':
        base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=1,
            decoder_dropout=0,
            activation=None
        )
    elif architecture.lower() == 'deeplabv3+':
        base_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=1,
            activation=None
        )
    elif architecture.lower() == 'fpn':
        base_model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=1,
            activation=None
        )
    elif architecture.lower() == 'pspnet':
        base_model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=1,
            activation=None
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Use 'unet', 'deeplabv3+', 'fpn', or 'pspnet'")
    
    # wrap base_model with UNetWithDropout
    model = UNetWithDropout(base_model, dropout_rate=dropout_rate)

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
        architecture=CONFIG['model'].get('architecture', 'unet'),
        device=DEVICE,
        dropout_rate=CONFIG['model'].get('dropout_rate', 0.3)
    )
    
   # Test with dummy input
    in_ch = get_input_channels(CONFIG['data']['modality'])
    dummy = torch.randn(2, in_ch, 128, 128).to(DEVICE)  # Batch of 2
    
    print(f"Testing forward pass:")
    print(f" Input shape: {dummy.shape}")
    
    # Set model to eval mode for testing
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    
    print(f" Output shape: {out.shape}")
    print(f" Output device: {out.device}")
    
    # Check output value range
    print(f" Output min: {out.min().item():.4f}")
    print(f" Output max: {out.max().item():.4f}")
    print(f"✓ Model works correctly on {DEVICE}!")
    print("="*60 + "\n")