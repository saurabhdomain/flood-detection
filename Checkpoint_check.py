23125# test_checkpoint.py
"""
Test loaded checkpoint model on validation set
No training - inference only
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Load config
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

from dataset import FloodDataset
from model import create_model

device = torch.device(CONFIG['training']['device'])
print(f"Using device: {device}")

# ============================================================
# LOAD CHECKPOINT (NO TRAINING)
# ============================================================

checkpoint_path = Path('./outputs/models/best_model.pt')

if checkpoint_path.exists():
    print(f"\n✓ Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model (MUST be same architecture)
    model = create_model(
        CONFIG['data']['modality'],
        CONFIG['model']['encoder_name'],
        device=device
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Evaluation mode
    
    print(f"✓ Model loaded successfully")
    print(f"  Epoch: {checkpoint.get('epoch', 0)}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
else:
    raise FileNotFoundError(f"✗ No checkpoint found: {checkpoint_path}")

# ============================================================
# VALIDATION DATASET & LOADER
# ============================================================

val_dataset = FloodDataset(
    CONFIG['data']['data_dir'],
    CONFIG['data']['modality'],
    CONFIG['data']['mask_source'],
    split='val'  # or 'test' if available
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['training']['batch_size'],
    shuffle=False,
    num_workers=CONFIG['training']['num_workers']
)

print(f"✓ Validation dataset: {len(val_dataset)} samples")

# ============================================================
# INFERENCE & METRICS
# ============================================================

criterion = nn.BCEWithLogitsLoss()
all_preds = []
all_targets = []

print("\n" + "="*70)
print("RUNNING VALIDATION INFERENCE")
print("="*70)

with torch.no_grad():
    val_loss = 0.0
    pbar = tqdm(val_loader, desc="Validation")
    
    for batch in pbar:
        images = batch['images'].to(device)
        masks = batch['masks'].to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        val_loss += loss.item()
        
        # Convert to binary predictions (threshold 0.5)
        preds = torch.sigmoid(outputs) > 0.5
        all_preds.extend(preds.cpu().numpy().flatten())
        all_targets.extend(masks.cpu().numpy().flatten())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

# Calculate metrics
avg_val_loss = val_loss / len(val_loader)
accuracy = accuracy_score(all_targets, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_targets, all_preds, average='binary', zero_division=0
)

print("\n" + "="*50)
print("VALIDATION RESULTS")
print("="*50)
print(f"Validation Loss:     {avg_val_loss:.4f}")
print(f"Accuracy:            {accuracy:.4f}")
print(f"Precision:           {precision:.4f}")
print(f"Recall:              {recall:.4f}")
print(f"F1-Score:            {f1:.4f}")
print("="*50)

print("\n✓ Checkpoint test complete!")
