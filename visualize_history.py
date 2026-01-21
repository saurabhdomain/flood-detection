#!/usr/bin/env python3
"""
Training History Visualization Tool

This script loads training history from training_history.json and provides
a detailed visualization with highlighting to help decide which epochs to keep.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# Configuration constants
OVERFITTING_WINDOW = 5  # Number of epochs to check for overfitting trend
OVERFITTING_THRESHOLD = 0.01  # Threshold for detecting overfitting
STABILITY_WINDOW = 5  # Number of epochs to check for stability
STABILITY_THRESHOLD = 0.01  # Standard deviation threshold for stable epochs

def load_history(history_path='training_history.json'):
    """Load training history from JSON file."""
    try:
        with open(history_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: History file not found: {history_path}")
        print(f"   Make sure the file exists or provide the correct path with --history")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in history file: {history_path}")
        print(f"   {str(e)}")
        exit(1)

def highlight_key_epochs(history):
    """Identify and return key epochs worth keeping."""
    epochs = history['epoch']
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    
    key_epochs = {
        'best_val': None,
        'best_train': None,
        'lr_changes': [],
        'overfitting_start': None,
        'stable_epochs': []
    }
    
    # Find best validation loss
    best_val_idx = np.argmin(val_losses)
    key_epochs['best_val'] = epochs[best_val_idx]
    
    # Find best training loss
    best_train_idx = np.argmin(train_losses)
    key_epochs['best_train'] = epochs[best_train_idx]
    
    # Find learning rate changes
    lrs = history['learning rate(lr)']
    for i in range(1, len(lrs)):
        if lrs[i] != lrs[i-1]:
            key_epochs['lr_changes'].append(epochs[i])
    
    # Detect overfitting (when val loss starts increasing while train keeps decreasing)
    for i in range(10, len(val_losses)):
        recent_val_trend = val_losses[i] - val_losses[i-OVERFITTING_WINDOW]
        recent_train_trend = train_losses[i] - train_losses[i-OVERFITTING_WINDOW]
        
        if recent_val_trend > OVERFITTING_THRESHOLD and recent_train_trend < 0:
            key_epochs['overfitting_start'] = epochs[i]
            break
    
    # Find stable epochs (low variation in last epochs)
    for i in range(STABILITY_WINDOW, len(val_losses)):
        window = val_losses[i-STABILITY_WINDOW:i]
        if np.std(window) < STABILITY_THRESHOLD:
            key_epochs['stable_epochs'].append(epochs[i])
    
    return key_epochs

def print_detailed_history(history):
    """Print detailed epoch-by-epoch breakdown with highlighting."""
    epochs = history['epoch']
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    lrs = history['learning rate(lr)']
    
    key_epochs = highlight_key_epochs(history)
    
    print("\n" + "="*80)
    print("TRAINING HISTORY ANALYSIS")
    print("="*80 + "\n")
    
    print("📊 KEY EPOCHS SUMMARY:")
    print("-" * 80)
    print(f"  🏆 Best Validation Loss: Epoch {key_epochs['best_val']} "
          f"(Val Loss: {val_losses[key_epochs['best_val']-1]:.4f})")
    print(f"  ⭐ Best Training Loss:   Epoch {key_epochs['best_train']} "
          f"(Train Loss: {train_losses[key_epochs['best_train']-1]:.4f})")
    
    if key_epochs['overfitting_start']:
        print(f"  ⚠️  Overfitting Started:   Epoch {key_epochs['overfitting_start']}")
    else:
        print(f"  ✅ No Clear Overfitting Detected")
    
    if key_epochs['lr_changes']:
        print(f"  📉 Learning Rate Changes: Epochs {key_epochs['lr_changes']}")
    
    print("\n" + "="*80)
    print("EPOCH-BY-EPOCH DETAILS")
    print("="*80)
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Gap':<12} {'LR':<12} {'Notes':<20}")
    print("-" * 80)
    
    for i, epoch in enumerate(epochs):
        train_loss = train_losses[i]
        val_loss = val_losses[i]
        lr = lrs[i]
        gap = val_loss - train_loss
        
        # Determine if epoch is highlighted
        notes = []
        highlight = ""
        
        if epoch == key_epochs['best_val']:
            notes.append("🏆 BEST VAL")
            highlight = ">>> "
        elif epoch == key_epochs['best_train']:
            notes.append("⭐ BEST TRAIN")
            highlight = ">>> "
        
        if epoch in key_epochs['lr_changes']:
            notes.append("📉 LR DROP")
            highlight = ">>> "
        
        if epoch == key_epochs['overfitting_start']:
            notes.append("⚠️  OVERFIT START")
            highlight = ">>> "
        
        if epoch in key_epochs['stable_epochs']:
            notes.append("✓ Stable")
        
        note_str = ", ".join(notes) if notes else ""
        
        print(f"{highlight}{epoch:<8} {train_loss:<15.6f} {val_loss:<15.6f} "
              f"{gap:<12.6f} {lr:<12.2e} {note_str:<20}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Provide recommendations
    best_epoch = key_epochs['best_val']
    print(f"\n💡 RECOMMENDED CHECKPOINTS TO KEEP:\n")
    print(f"   1. Epoch {best_epoch} - Best validation loss (PRIMARY)")
    
    if key_epochs['overfitting_start']:
        overfit_epoch = key_epochs['overfitting_start']
        if overfit_epoch > best_epoch:
            print(f"   2. Epoch {overfit_epoch-1} - Last good epoch before overfitting")
    
    for lr_change in key_epochs['lr_changes'][:2]:
        print(f"   3. Epoch {lr_change} - Learning rate adjustment checkpoint")
    
    print(f"\n❌ CHECKPOINTS TO CONSIDER DISCARDING:\n")
    print(f"   • Early epochs (1-10) - Still warming up")
    
    if key_epochs['overfitting_start']:
        overfit_epoch = key_epochs['overfitting_start']
        print(f"   • Epochs after {overfit_epoch} - Overfitting territory")
    
    # Find epochs with high validation loss
    high_val_epochs = [i+1 for i, v in enumerate(val_losses) if v > np.percentile(val_losses, 75)]
    if high_val_epochs:
        print(f"   • Epochs with high val loss: {high_val_epochs[:5]}")
    
    print("\n")

def plot_history(history, save_path=None):
    """Create comprehensive visualization of training history."""
    epochs = history['epoch']
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    lrs = history['learning rate(lr)']
    
    key_epochs = highlight_key_epochs(history)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Training and Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    
    # Highlight best validation epoch
    best_val_idx = key_epochs['best_val'] - 1
    ax1.scatter(key_epochs['best_val'], val_losses[best_val_idx], 
                color='gold', s=200, marker='*', zorder=5, 
                label=f'Best Val (Epoch {key_epochs["best_val"]})')
    
    # Highlight overfitting start
    if key_epochs['overfitting_start']:
        overfit_idx = key_epochs['overfitting_start'] - 1
        ax1.axvline(key_epochs['overfitting_start'], color='orange', 
                   linestyle='--', alpha=0.7, label='Overfitting Start')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves with Key Epochs Highlighted', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss Gap (Overfitting indicator)
    ax2 = axes[0, 1]
    gaps = [v - t for v, t in zip(val_losses, train_losses)]
    colors = ['red' if g > 0.05 else 'green' for g in gaps]
    ax2.bar(epochs, gaps, color=colors, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, 
                label='Overfitting Threshold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Val Loss - Train Loss', fontsize=12)
    ax2.set_title('Overfitting Gap Analysis', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Learning Rate Schedule
    ax3 = axes[1, 0]
    ax3.plot(epochs, lrs, 'g-o', linewidth=2, markersize=6)
    for lr_change in key_epochs['lr_changes']:
        ax3.axvline(lr_change, color='purple', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Epoch Recommendations
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    recommendations = []
    recommendations.append("📌 EPOCHS TO KEEP:\n")
    recommendations.append(f"  • Epoch {key_epochs['best_val']} (Best Val Loss)\n")
    
    if key_epochs['overfitting_start']:
        recommendations.append(f"  • Epoch {key_epochs['overfitting_start']-1} (Pre-overfit)\n")
    
    recommendations.append("\n❌ EPOCHS TO DISCARD:\n")
    recommendations.append(f"  • Epochs 1-10 (Warmup)\n")
    
    if key_epochs['overfitting_start']:
        recommendations.append(f"  • Epochs {key_epochs['overfitting_start']}+ (Overfitting)\n")
    
    recommendations.append("\n📊 STATISTICS:\n")
    recommendations.append(f"  • Total Epochs: {len(epochs)}\n")
    recommendations.append(f"  • Best Val Loss: {min(val_losses):.4f}\n")
    recommendations.append(f"  • Best Train Loss: {min(train_losses):.4f}\n")
    recommendations.append(f"  • Final Gap: {val_losses[-1] - train_losses[-1]:.4f}\n")
    
    recommendation_text = "".join(recommendations)
    ax4.text(0.1, 0.5, recommendation_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax4.set_title('Recommendations', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(
        description='Visualize and analyze training history to decide which epochs to keep'
    )
    parser.add_argument(
        '--history',
        type=str,
        default='training_history.json',
        help='Path to training history JSON file (default: training_history.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='training_history_analysis.png',
        help='Output path for visualization plot (default: training_history_analysis.png)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the plot interactively'
    )
    
    args = parser.parse_args()
    
    # Load history
    print(f"Loading training history from: {args.history}")
    history = load_history(args.history)
    
    # Print detailed analysis
    print_detailed_history(history)
    
    # Create visualization
    print(f"\nGenerating visualization...")
    fig = plot_history(history, save_path=args.output)
    
    if args.show:
        plt.show()
    else:
        plt.close()
    
    print("\n✅ Analysis complete!")
    print(f"   View the detailed analysis above")
    print(f"   Check the visualization: {args.output}\n")

if __name__ == '__main__':
    main()
