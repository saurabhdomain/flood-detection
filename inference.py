"""
inference.py - Inference pipeline for flood detection
Load trained model and make predictions on new data
"""

import torch
import yaml
import argparse
import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
from model import create_model


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path, config):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
    
    Returns:
        Loaded model
    """
    device = torch.device(config['training']['device'])
    
    # Create model
    model = create_model(
        config['data']['modality'],
        config['model']['encoder_name'],
        architecture=config['model'].get('architecture', 'unet'),
        device=device,
        dropout_rate=config['model'].get('dropout_rate', 0.5)
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    return model, device


def load_image(image_path, modality='s1'):
    """
    Load and preprocess image for inference
    
    Args:
        image_path: Path to image file
        modality: Data modality (s1, s2, s1_s2)
    
    Returns:
        Preprocessed image tensor
    """
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)
        
        if modality == 's1':
            # SAR preprocessing
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
            image = np.clip(image, -1e6, 1e6)
            image = np.log1p(np.abs(image))
            mean = image.mean(axis=(1, 2), keepdims=True)
            std = image.std(axis=(1, 2), keepdims=True)
            std = np.maximum(std, 0.01)
            image = (image - mean) / std
            image = np.clip(image, -10, 10)
        else:
            # Optical preprocessing
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            image = image / 10000.0
            image = np.clip(image, 0, 1)
    
    return torch.from_numpy(image).float()


def predict(model, image, device):
    """
    Make prediction on image
    
    Args:
        model: Trained model
        image: Input image tensor
        device: Device to use
    
    Returns:
        Prediction mask
    """
    with torch.no_grad():
        # Add batch dimension
        image = image.unsqueeze(0).to(device)
        
        # Forward pass
        output = model(image)
        
        # Apply sigmoid and threshold
        prediction = torch.sigmoid(output)
        prediction = (prediction > 0.5).float()
        
        # Remove batch dimension
        prediction = prediction.squeeze(0).squeeze(0)
    
    return prediction.cpu().numpy()


def visualize_prediction(image, prediction, save_path=None):
    """
    Visualize prediction
    
    Args:
        image: Input image (numpy array)
        prediction: Prediction mask (numpy array)
        save_path: Optional path to save visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input image (first channel)
    axes[0].imshow(image[0], cmap='gray')
    axes[0].set_title('Input Image (Channel 1)')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(prediction, cmap='Blues')
    axes[1].set_title('Flood Prediction')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image[0], cmap='gray')
    axes[2].imshow(prediction, cmap='Reds', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_path}")
    else:
        plt.show()


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Flood Detection Inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='outputs/models/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output visualization')
    parser.add_argument('--save-mask', type=str, default=None,
                        help='Path to save prediction mask as GeoTIFF')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Flood Detection Inference")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_config(args.config)
    
    # Load model
    model, device = load_model(args.checkpoint, config)
    
    # Load image
    print(f"Loading image: {args.image}")
    image = load_image(args.image, config['data']['modality'])
    print(f"✓ Image shape: {image.shape}")
    
    # Make prediction
    print("Making prediction...")
    prediction = predict(model, image, device)
    print(f"✓ Prediction shape: {prediction.shape}")
    
    # Calculate flood statistics
    flood_pixels = prediction.sum()
    total_pixels = prediction.size
    flood_percentage = (flood_pixels / total_pixels) * 100
    
    print(f"\n📊 Flood Statistics:")
    print(f"  Flooded pixels: {int(flood_pixels):,}")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Flood coverage: {flood_percentage:.2f}%")
    
    # Save mask if requested
    if args.save_mask:
        output_path = Path(args.save_mask)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy geotransform from input
        with rasterio.open(args.image) as src:
            profile = src.profile.copy()
            profile.update({
                'count': 1,
                'dtype': 'uint8',
                'compress': 'lzw'
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write((prediction * 255).astype(np.uint8), 1)
        
        print(f"✓ Mask saved to: {args.save_mask}")
    
    # Visualize
    visualize_prediction(
        image.numpy(),
        prediction,
        save_path=args.output
    )
    
    print(f"\n{'='*60}")
    print("✓ Inference complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
