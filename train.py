import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import os
import json
import numpy as np

script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

# load config
with open('config.yaml','r') as f:
    CONFIG = yaml.safe_load(f)

# import modules 

from dataset import FloodDataset
from model import create_model

# setup device
Device = torch.device(CONFIG['training']['device'])
print(f"\n🖥️  Using device: {Device}")

# create dataset and dataloader

print('Creating Dataset....')

training_dataset = FloodDataset(CONFIG['data']['data_dir'],CONFIG['data']['modality'],CONFIG['data']['mask_source'],split = 'train')

val_dataset = FloodDataset(CONFIG['data']['data_dir'],CONFIG['data']['modality'],CONFIG['data']['mask_source'],split = 'val')

train_loader = DataLoader(training_dataset,CONFIG['training']['batch_size'],shuffle = True,num_workers = CONFIG['training']['num_workers'])

val_loader = DataLoader(val_dataset,CONFIG['training']['batch_size'],shuffle = False,num_workers = CONFIG['training']['num_workers'])



# create Model

print('Creating Model....')

model = create_model(CONFIG['data']['modality'],CONFIG['model']['encoder_name'],
                     device= Device,dropout_rate= CONFIG['model'].get('dropout_rate',0.3))

print(f" train batch {len(train_loader)}")
print(f" validation  batch {len(val_loader)}")
print(f"\n📊 Training Configuration Check:")
print(f"  Training samples: {len(training_dataset)}")
print(f"  Validation samples: {len(val_dataset)}")
print(f"  Batch size: {CONFIG['training']['batch_size']}")
print(f"  Learning rate: {CONFIG['training']['learning_rate']}")
print(f"  Weight decay: {CONFIG['training'].get('weight_decay', 'NOT SET')}")
print(f"  Augmentation: {CONFIG['data'].get('augmentation', False)}")
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test augmentation variability
print(f"\n🧪 Testing augmentation variability:")
sample1 = training_dataset[0]
sample2 = training_dataset[0]  # Same index
diff = abs(sample1['images'].mean() - sample2['images'].mean())
print(f"  Sample mean difference: {diff:.6f}")
if diff > 0.001:
    print("  ✅ Augmentation is working!")
else:
    print("  ⚠️  Augmentation may not be working or too weak!")

# Setup Training
class SmoothBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        # Apply label smoothing: 0 -> 0.05, 1 -> 0.95
        target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(pred, target_smooth)

# criterion = nn.BCEWithLogitsLoss()  # ← OLD
criterion = SmoothBCEWithLogitsLoss(smoothing=0.1)

optimiser = torch.optim.Adam(model.parameters(),lr = CONFIG['training']['learning_rate'], weight_decay= 1e-4)

scheduler = torch .optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode= 'min', factor =0.5 , patience= 5,min_lr=1e-6,)
# create output directorry

output_dir = Path(CONFIG['output']['checkpoint_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

print(f"✓ Loss: BCEWithLogitsLoss")
print(f"✓ Optimizer: Adam (lr={CONFIG['training']['learning_rate']})\n")
print(f"✓ Scheduler: ReduceLROnPlateau (patience=3)\n")

# training loop with history tracking

best_val_loss = float('inf')
patience_counter =0
max_patience = 15

history={
    'epoch' :[],
    'train_losses':[],
    'val_losses':[],
    'learning rate(lr)' :[]
}


for epoch in range(CONFIG['training']['num_epochs']):

    model.train()
    train_loss = 0.0
    val_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['training']['num_epochs']} - Training", unit="batch")

    batch_losses = []
    # taking images from train in batches
    for batch in pbar:

        images = batch['images'].to(Device)
        masks = batch['masks'].to(Device)

        # Check for anomalies in input data
        if images.isnan().any() or images.isinf().any():
            print(f"⚠️  NaN/Inf in input images at batch {len(batch_losses)}")
            continue
        
        optimiser.zero_grad()
        # foward pass 
        outputs = model(images)

        loss = criterion(outputs, masks)

        # Check for invalid values
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  WARNING: Invalid loss detected: {loss.item()}")
            print(f"  Images - min: {images.min()}, max: {images.max()}")
            print(f"  Outputs - min: {outputs.min()}, max: {outputs.max()}")
            print(f"  Masks - min: {masks.min()}, max: {masks.max()}")
            continue  # Skip this batch

        
        batch_losses.append(loss.item())

        # Check for unstable loss
        if len(batch_losses) > 10:
            recent_mean = np.mean(batch_losses[-10:])
            recent_std = np.std(batch_losses[-10:])
            
            if loss.item() > recent_mean + 3*recent_std:
                print(f"⚠️  Unstable loss spike: {loss.item():.4f} (mean: {recent_mean:.4f})")
        

        #backward

      
        loss.backward()
        # ✅ Check gradient norms BEFORE clipping
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
         # ✅ ADD GRADIENT CLIPPING HERE (before optimizer.step())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        train_loss +=loss.item()
        pbar.set_postfix({'loss': loss.item()})

    avg_train_loss = train_loss / len(train_loader)

    # validation loop

    model.eval()
    valid_batches = 0 
    with torch.no_grad():
        pbar = tqdm(val_loader,desc=f"EPOCH {epoch+1}/{CONFIG['training']['num_epochs']} [VAL]",unit="batch")

        for batch in pbar :
            images = batch['images'].to(Device)
            masks = batch['masks'].to(Device)

            # ✅ ADD: Skip batches with invalid data
            if images.isnan().any() or images.isinf().any():
                print(f"⚠️  Skipping validation batch with invalid input")
                continue

            output = model(images)
            loss = criterion(output, masks)

            # ✅ ADD: Skip batches with invalid loss
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 10:
                print(f"⚠️  Skipping validation batch with extreme loss: {loss.item():.4f}")
                continue

            val_loss += loss.item()
            valid_batches += 1  # ✅ ADD
            pbar.set_postfix({'val_loss': loss.item()})
    
    # ✅ FIX: Use valid_batches instead of len(val_loader)
    avg_val_loss = val_loss / max(valid_batches, 1)


    # track history
    history['epoch'].append(epoch +1)
    history['train_losses'].append(avg_train_loss)
    history['val_losses'].append(avg_val_loss)
    history['learning rate(lr)'].append(optimiser.param_groups[0]['lr'])

    # ---- PRINT RESULTS ----
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    print(f"  LR:         {optimiser.param_groups[0]['lr']:.6f}")
    print(f"  Gap:        {avg_val_loss - avg_train_loss:.4f}")

# Update learning rate based on validation loss

    scheduler.step(avg_val_loss)
    #----- save checkpoint -----

    if avg_val_loss<best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter =0
        checkpoint_path = output_dir / f"best_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'loss':avg_val_loss,
            'config': CONFIG
        }, checkpoint_path)

        print(f"  ✓ Saved best model: {checkpoint_path}")
    else:
        patience_counter +=1
        print(f" ! No improvement in validation loss. Patience counter: {patience_counter}/{max_patience}")

        # early stopping
        if patience_counter>= max_patience:
            print(f"\n Early stopping at epoch {epoch +1}")
            break

    if (epoch + 1) % 5 == 0:
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  ✓ Saved checkpoint: {checkpoint_path}")
    
    print()


# Save training history
with open(output_dir / 'training_history.json', 'w') as f:
    json.dump(history, f, indent=2)


print("\n" + "="*60)
print("✓ TRAINING COMPLETE!")
print(f"✓ Best model saved to: {output_dir}/best_model.pt")
print("="*60 + "\n")




