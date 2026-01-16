"""
compare_experiments.py - Compare different training experiments
Analyze training histories and identify overfitting patterns
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_history(history_path):
    """Load training history from JSON file"""
    with open(history_path, 'r') as f:
        return json.load(f)


def analyze_overfitting(history):
    """
    Analyze training history for overfitting indicators
    
    Args:
        history: Training history dictionary
    
    Returns:
        Dictionary of analysis results
    """
    train_loss = np.array(history.get('train_losses', history.get('train_loss', [])))
    val_loss = np.array(history.get('val_losses', history.get('val_loss', [])))
    
    if len(train_loss) == 0 or len(val_loss) == 0:
        return None
    
    # Calculate gaps
    gaps = val_loss - train_loss
    
    # Analysis metrics
    analysis = {
        'initial_train_loss': train_loss[0],
        'initial_val_loss': val_loss[0],
        'final_train_loss': train_loss[-1],
        'final_val_loss': val_loss[-1],
        'train_improvement': train_loss[0] - train_loss[-1],
        'val_improvement': val_loss[0] - val_loss[-1],
        'avg_gap': np.mean(gaps),
        'final_gap': gaps[-1],
        'max_gap': np.max(gaps),
        'best_val_loss': np.min(val_loss),
        'best_val_epoch': np.argmin(val_loss) + 1,
        'val_std_last10': np.std(val_loss[-10:]) if len(val_loss) >= 10 else 0,
    }
    
    # Overfitting indicators
    analysis['overfitting_score'] = 0
    
    if analysis['avg_gap'] > 0.05:
        analysis['overfitting_score'] += 2
        analysis['overfitting_reason'] = "Large train-val gap"
    
    if analysis['train_improvement'] > 0.1 and analysis['val_improvement'] < 0.05:
        analysis['overfitting_score'] += 2
        analysis['overfitting_reason'] = "Train loss decreases but val loss doesn't"
    
    if analysis['val_std_last10'] > 0.02:
        analysis['overfitting_score'] += 1
        analysis['overfitting_reason'] = "High validation loss variance"
    
    # Classification
    if analysis['overfitting_score'] >= 3:
        analysis['status'] = "Severe Overfitting"
    elif analysis['overfitting_score'] >= 2:
        analysis['status'] = "Moderate Overfitting"
    elif analysis['overfitting_score'] >= 1:
        analysis['status'] = "Mild Overfitting"
    else:
        analysis['status'] = "Good Generalization"
    
    return analysis


def plot_training_history(history, title="Training History", save_path=None):
    """
    Plot training and validation losses
    
    Args:
        history: Training history dictionary
        title: Plot title
        save_path: Path to save the plot
    """
    train_loss = history.get('train_losses', history.get('train_loss', []))
    val_loss = history.get('val_losses', history.get('val_loss', []))
    epochs = history.get('epoch', list(range(1, len(train_loss) + 1)))
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gap plot
    plt.subplot(1, 2, 2)
    gaps = np.array(val_loss) - np.array(train_loss)
    plt.plot(epochs, gaps, 'g-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss - Train Loss')
    plt.title(f'{title} - Generalization Gap')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()


def print_analysis(analysis, name="Experiment"):
    """Print analysis results in a formatted way"""
    print(f"\n{'='*60}")
    print(f"{name} Analysis")
    print(f"{'='*60}")
    
    if analysis is None:
        print("❌ No data available for analysis")
        return
    
    print(f"\n📊 Loss Summary:")
    print(f"  Initial - Train: {analysis['initial_train_loss']:.4f}, Val: {analysis['initial_val_loss']:.4f}")
    print(f"  Final   - Train: {analysis['final_train_loss']:.4f}, Val: {analysis['final_val_loss']:.4f}")
    print(f"  Best Val Loss: {analysis['best_val_loss']:.4f} (Epoch {analysis['best_val_epoch']})")
    
    print(f"\n📈 Improvement:")
    print(f"  Train: {analysis['train_improvement']:.4f}")
    print(f"  Val:   {analysis['val_improvement']:.4f}")
    
    print(f"\n🎯 Generalization Gap:")
    print(f"  Average: {analysis['avg_gap']:.4f}")
    print(f"  Final:   {analysis['final_gap']:.4f}")
    print(f"  Maximum: {analysis['max_gap']:.4f}")
    
    print(f"\n📊 Validation Stability:")
    print(f"  Std Dev (last 10): {analysis['val_std_last10']:.4f}")
    
    print(f"\n⚠️  Overfitting Assessment:")
    print(f"  Status: {analysis['status']}")
    print(f"  Score:  {analysis['overfitting_score']}/5")
    
    if analysis['overfitting_score'] > 0:
        print(f"  Reason: {analysis.get('overfitting_reason', 'Multiple factors')}")
    
    print(f"\n{'='*60}\n")


def compare_histories(histories, names):
    """
    Compare multiple training histories
    
    Args:
        histories: List of history dictionaries
        names: List of experiment names
    """
    print(f"\n{'='*60}")
    print("Experiment Comparison")
    print(f"{'='*60}\n")
    
    analyses = []
    for history, name in zip(histories, names):
        analysis = analyze_overfitting(history)
        analyses.append(analysis)
        print(f"{name}:")
        if analysis:
            print(f"  Best Val Loss: {analysis['best_val_loss']:.4f}")
            print(f"  Final Gap: {analysis['final_gap']:.4f}")
            print(f"  Status: {analysis['status']}")
        else:
            print(f"  No data available")
        print()
    
    # Find best experiment
    valid_analyses = [(a, n) for a, n in zip(analyses, names) if a is not None]
    if valid_analyses:
        best = min(valid_analyses, key=lambda x: x[0]['best_val_loss'])
        print(f"🏆 Best Experiment: {best[1]}")
        print(f"   Best Val Loss: {best[0]['best_val_loss']:.4f}")
        print(f"   Status: {best[0]['status']}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Compare training experiments')
    parser.add_argument('--history', type=str, default='training_history.json',
                        help='Path to training history file')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots')
    parser.add_argument('--compare', nargs='+', type=str,
                        help='Compare multiple history files')
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple experiments
        histories = []
        names = []
        for path in args.compare:
            if Path(path).exists():
                histories.append(load_history(path))
                names.append(Path(path).parent.name)
            else:
                print(f"⚠️  File not found: {path}")
        
        if histories:
            compare_histories(histories, names)
            
            if args.plot:
                for history, name in zip(histories, names):
                    plot_training_history(history, title=name)
    else:
        # Single experiment analysis
        if not Path(args.history).exists():
            print(f"❌ File not found: {args.history}")
            return
        
        history = load_history(args.history)
        analysis = analyze_overfitting(history)
        print_analysis(analysis, name=Path(args.history).parent.name)
        
        if args.plot:
            plot_training_history(
                history,
                title="Training History",
                save_path="training_analysis.png"
            )


if __name__ == "__main__":
    # If run without arguments, analyze the current training history
    import sys
    if len(sys.argv) == 1:
        history_path = Path("training_history.json")
        if history_path.exists():
            print("Analyzing current training history...")
            history = load_history(history_path)
            analysis = analyze_overfitting(history)
            print_analysis(analysis)
        else:
            print("No training_history.json found in current directory.")
            print("Usage: python compare_experiments.py --history path/to/history.json")
    else:
        main()
