"""
pipeline.py - Main MLOps pipeline orchestrator
Orchestrates training, evaluation, and model management
"""

import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np
import os

# Import project modules
from dataset import FloodDataset
from model import create_model
from metrics import MetricsTracker
from experiment_tracker import ExperimentTracker


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class SmoothBCEWithLogitsLoss(nn.Module):
    """BCE Loss with label smoothing"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(pred, target_smooth)


class FloodDetectionPipeline:
    """
    Main pipeline for flood detection training and evaluation
    """
    def __init__(self, config_path='config.yaml', experiment_name=None):
        """
        Initialize pipeline
        
        Args:
            config_path: Path to configuration file
            experiment_name: Custom experiment name
        """
        self.config = load_config(config_path)
        self.device = torch.device(self.config['training']['device'])
        
        # Setup experiment tracking
        if experiment_name is None:
            experiment_name = f"{self.config['mlops']['experiment_name']}_{self.config['data']['modality']}_{self.config['model']['encoder_name']}"
        
        self.tracker = ExperimentTracker(
            experiment_name=experiment_name,
            tracking_uri=self.config['mlops'].get('mlflow_tracking_uri', './mlruns'),
            use_mlflow=self.config['mlops'].get('use_mlflow', True)
        )
        
        # Log configuration
        self.tracker.log_config(self.config)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        self.patience_counter = 0
        
        print(f"\n{'='*60}")
        print(f"🚀 Flood Detection MLOps Pipeline")
        print(f"{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
    
    def setup_data(self):
        """Setup datasets and dataloaders"""
        print("📊 Setting up datasets...")
        
        # Create datasets
        train_dataset = FloodDataset(
            self.config['data']['data_dir'],
            self.config['data']['modality'],
            self.config['data']['mask_source'],
            split='train'
        )
        
        val_dataset = FloodDataset(
            self.config['data']['data_dir'],
            self.config['data']['modality'],
            self.config['data']['mask_source'],
            split='val'
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers']
        )
        
        print(f"✓ Training samples: {len(train_dataset)}")
        print(f"✓ Validation samples: {len(val_dataset)}")
        print(f"✓ Training batches: {len(self.train_loader)}")
        print(f"✓ Validation batches: {len(self.val_loader)}\n")
    
    def setup_model(self):
        """Setup model, loss, optimizer, and scheduler"""
        print("🏗️  Setting up model...")
        
        # Create model
        self.model = create_model(
            self.config['data']['modality'],
            self.config['model']['encoder_name'],
            device=self.device,
            dropout_rate=self.config['model'].get('dropout_rate', 0.5)
        )
        
        # Setup loss function
        smoothing = self.config['model'].get('label_smoothing', 0.1)
        self.criterion = SmoothBCEWithLogitsLoss(smoothing=smoothing)
        print(f"✓ Loss: BCEWithLogitsLoss (smoothing={smoothing})")
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 1e-4)
        )
        print(f"✓ Optimizer: Adam (lr={self.config['training']['learning_rate']})")
        
        # Setup scheduler
        scheduler_config = self.config['training'].get('scheduler', {})
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5),
            min_lr=scheduler_config.get('min_lr', 1e-6)
        )
        print(f"✓ Scheduler: ReduceLROnPlateau\n")
        
        # Log model info
        n_params = sum(p.numel() for p in self.model.parameters())
        self.tracker.log_params({
            'model_parameters': n_params,
            'encoder': self.config['model']['encoder_name'],
            'modality': self.config['data']['modality']
        })
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average training loss
        """
        self.model.train()
        train_loss = 0.0
        train_metrics = MetricsTracker()
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']} [TRAIN]",
            unit="batch"
        )
        
        for batch in pbar:
            images = batch['images'].to(self.device)
            masks = batch['masks'].to(self.device)
            
            # Skip invalid data
            if images.isnan().any() or images.isinf().any():
                continue
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            clip_norm = self.config['training'].get('gradient_clip_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
            
            self.optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_metrics.update(outputs, masks)
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(self.train_loader)
        train_metrics_dict = train_metrics.get_metrics()
        
        return avg_train_loss, train_metrics_dict
    
    def validate(self, epoch):
        """
        Validate the model
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Tuple of (average validation loss, validation metrics)
        """
        self.model.eval()
        val_loss = 0.0
        val_metrics = MetricsTracker()
        
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']} [VAL]",
            unit="batch"
        )
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['images'].to(self.device)
                masks = batch['masks'].to(self.device)
                
                # Skip invalid data
                if images.isnan().any() or images.isinf().any():
                    continue
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Skip invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                val_loss += loss.item()
                val_metrics.update(outputs, masks)
                
                pbar.set_postfix({'val_loss': loss.item()})
        
        avg_val_loss = val_loss / len(self.val_loader)
        val_metrics_dict = val_metrics.get_metrics()
        
        return avg_val_loss, val_metrics_dict
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
        """
        output_dir = Path(self.config['output']['checkpoint_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if is_best:
            checkpoint_path = output_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            self.tracker.log_checkpoint(checkpoint, "best_model")
            print(f"  ✓ Saved best model: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        print("🎯 Starting training...\n")
        
        # Setup
        self.setup_data()
        self.setup_model()
        
        # Training history
        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_dice': [],
            'learning_rate': []
        }
        
        max_patience = self.config['training'].get('early_stopping_patience', 15)
        primary_metric = self.config['mlops'].get('primary_metric', 'val_iou')
        
        for epoch in range(self.config['training']['num_epochs']):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            all_metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            
            self.tracker.log_metrics(all_metrics, step=epoch)
            
            # Update history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_iou'].append(val_metrics['iou'])
            history['val_dice'].append(val_metrics['dice'])
            history['learning_rate'].append(current_lr)
            
            # Print summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | IoU: {train_metrics['iou']:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | IoU: {val_metrics['iou']:.4f}")
            print(f"  Val Dice:   {val_metrics['dice']:.4f} | F1: {val_metrics['f1_score']:.4f}")
            print(f"  LR:         {current_lr:.6f}")
            print(f"  Gap:        {val_loss - train_loss:.4f}")
            
            # Check for best model
            current_metric = val_metrics['iou']
            is_best = current_metric > self.best_val_metric
            
            if is_best:
                self.best_val_metric = current_metric
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, all_metrics, is_best=True)
            else:
                self.patience_counter += 1
                print(f"  ! No improvement. Patience: {self.patience_counter}/{max_patience}")
            
            # Early stopping
            if self.patience_counter >= max_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
            
            print()
        
        # Save history
        output_dir = Path(self.config['output']['checkpoint_dir'])
        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        self.tracker.log_artifact(str(output_dir / 'training_history.json'))
        
        print(f"\n{'='*60}")
        print("✓ TRAINING COMPLETE!")
        print(f"  Best Val IoU: {self.best_val_metric:.4f}")
        print(f"  Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        # End tracking
        self.tracker.end_run()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Flood Detection MLOps Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment name')
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = FloodDetectionPipeline(
        config_path=args.config,
        experiment_name=args.experiment
    )
    
    pipeline.train()


if __name__ == "__main__":
    main()
