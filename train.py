import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import os
import json

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


print(f" train batch {len(train_loader)}")
print(f" validation  batch {len(val_loader)}")

# create Model

print('Creating Model....')

model = create_model(CONFIG['data']['modality'],CONFIG['model']['encoder_name'],
                     device= Device,dropout_rate= CONFIG['model'].get('dropout_rate',0.3))


# Setup Training

criterion = nn.BCEWithLogitsLoss()

optimiser = torch.optim.Adam(model.parameters(),lr = CONFIG['training']['learning_rate'], weight_decay= 1e-5)

scheduler = torch .optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode= 'min', factor =0.5 , patience= 3)
# create output directorry

output_dir = Path(CONFIG['output']['checkpoint_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

print(f"✓ Loss: BCEWithLogitsLoss")
print(f"✓ Optimizer: Adam (lr={CONFIG['training']['learning_rate']})\n")
print(f"✓ Scheduler: ReduceLROnPlateau (patience=3)\n")

# training loop with history tracking

best_val_loss = float('inf')
patience_counter =0
max_patience = 5

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


    # taking images from train in batches
    for batch in pbar:

        images = batch['images'].to(Device)
        masks = batch['masks'].to(Device)

        # foward pass 
        outputs = model(images)

        loss = criterion(outputs, masks)

        #backward

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        train_loss +=loss.item()
        pbar.set_postfix({'loss': loss.item()})

    avg_train_loss = train_loss / len(train_loader)

    # validation loop

    model.eval()

    with torch.no_grad():
        pbar = tqdm(val_loader,desc=f"EPOCH {epoch+1}/{CONFIG['training']['num_epochs']} [VAL]",unit="batch")

        for batch in pbar :
            images = batch['images'].to(Device)
            masks = batch['masks'].to(Device)

            output = model(images)
            loss = criterion(output, masks)

            val_loss += loss.item()
            pbar.set_postfix({'val_loss': loss.item()})
    
    avg_val_loss = val_loss/ len(val_loader)


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




