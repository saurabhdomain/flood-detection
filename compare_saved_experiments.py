import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_experiment(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_metrics(experiments):
    print(f"\n{'='*80}")
    print(f"{'EXPERIMENT COMPARISON':^80}")
    print(f"{'='*80}")
    
    # Create comparison table
    data = []
    for exp in experiments:
        summary = exp.get('summary', {})
        data.append({
            'Name': exp['experiment_name'],
            'Best Val Loss': summary.get('best_val_loss', 'N/A'),
            'Best Val Epoch': summary.get('best_val_epoch', 'N/A'),
            'Final Train Loss': summary.get('final_train_loss', 'N/A'),
            'Final Gap': summary.get('avg_generalization_gap', 'N/A'),
            'Total Epochs': summary.get('total_epochs', 'N/A')
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print(f"{'='*80}\n")
    
    return df

def plot_comparisons(experiments, save_path='comparison_plot.png'):
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Validation Loss
    plt.subplot(1, 2, 1)
    for exp in experiments:
        history = exp.get('training_history', {})
        val_loss = history.get('val_losses', [])
        epochs = history.get('epoch', list(range(1, len(val_loss)+1)))
        
        # Determine strict name for legend
        name = exp['experiment_name']
        if len(name) > 20: name = name[:17] + "..."
        
        plt.plot(epochs, val_loss, label=f"{name} (Best: {min(val_loss):.4f})")
    
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Train Loss
    plt.subplot(1, 2, 2)
    for exp in experiments:
        history = exp.get('training_history', {})
        train_loss = history.get('train_losses', [])
        epochs = history.get('epoch', list(range(1, len(train_loss)+1)))
        name = exp['experiment_name']
        if len(name) > 20: name = name[:17] + "..."
            
        plt.plot(epochs, train_loss, label=name)
        
    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Comparison plot saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare saved experiments")
    parser.add_argument("--files", nargs='+', required=True, help="JSON files to compare")
    args = parser.parse_args()
    
    experiments = []
    for file in args.files:
        try:
            exp = load_experiment(file)
            experiments.append(exp)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if experiments:
        compare_metrics(experiments)
        plot_comparisons(experiments)
    else:
        print("No valid experiments found.")
