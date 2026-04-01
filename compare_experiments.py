"""
compare_experiments.py - Comprehensive experiment comparison tool
Analyze training histories, detect overfitting, and compare experiments
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def load_history(history_path):
    """Load training history from JSON file"""
    with open(history_path, 'r') as f:
        return json.load(f)


def load_saved_experiment(filepath):
    """Load saved experiment file (from save_experiment.py)"""
    with open(filepath, 'r') as f:
        return json.load(f)


def detect_file_type(filepath):
    """
    Detect if file is raw history or saved experiment
    
    Returns:
        'raw' for training_history.json format
        'saved' for save_experiment.py format
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if 'experiment_name' in data and 'training_history' in data:
        return 'saved', data
    else:
        return 'raw', data


def normalize_history(filepath):
    """
    Load file and normalize to consistent format
    
    Returns:
        Tuple of (name, history_dict, metadata)
    """
    file_type, data = detect_file_type(filepath)
    
    if file_type == 'saved':
        name = data.get('experiment_name', Path(filepath).stem)
        history = data.get('training_history', {})
        metadata = data.get('summary', {})
        
        # Normalize key names
        if 'train_losses' in history:
            history['train_loss'] = history.pop('train_losses')
        if 'val_losses' in history:
            history['val_loss'] = history.pop('val_losses')
    else:
        name = Path(filepath).stem
        history = data
        metadata = {}
        
        # Normalize key names
        if 'train_losses' in history:
            history['train_loss'] = history.pop('train_losses')
        if 'val_losses' in history:
            history['val_loss'] = history.pop('val_losses')
    
    return name, history, metadata


def analyze_overfitting(history):
    """
    Analyze training history for overfitting indicators
    
    Args:
        history: Training history dictionary
    
    Returns:
        Dictionary of analysis results
    """
    train_loss = np.array(history.get('train_loss', history.get('train_losses', [])))
    val_loss = np.array(history.get('val_loss', history.get('val_losses', [])))
    
    if len(train_loss) == 0 or len(val_loss) == 0:
        return None
    
    # Calculate gaps
    gaps = val_loss - train_loss
    
    # Analysis metrics
    analysis = {
        'initial_train_loss': float(train_loss[0]),
        'initial_val_loss': float(val_loss[0]),
        'final_train_loss': float(train_loss[-1]),
        'final_val_loss': float(val_loss[-1]),
        'train_improvement': float(train_loss[0] - train_loss[-1]),
        'val_improvement': float(val_loss[0] - val_loss[-1]),
        'avg_gap': float(np.mean(gaps)),
        'final_gap': float(gaps[-1]),
        'max_gap': float(np.max(gaps)),
        'best_val_loss': float(np.min(val_loss)),
        'best_val_epoch': int(np.argmin(val_loss) + 1),
        'total_epochs': len(train_loss),
        'val_std_last10': float(np.std(val_loss[-10:])) if len(val_loss) >= 10 else 0,
    }
    
    # IoU/Dice if available
    if 'val_iou' in history:
        analysis['best_val_iou'] = float(max(history['val_iou']))
    if 'val_dice' in history:
        analysis['best_val_dice'] = float(max(history['val_dice']))
    
    # Overfitting indicators
    analysis['overfitting_score'] = 0
    analysis['overfitting_reasons'] = []
    
    if analysis['avg_gap'] > 0.05:
        analysis['overfitting_score'] += 2
        analysis['overfitting_reasons'].append("Large train-val gap")
    
    if analysis['train_improvement'] > 0.1 and analysis['val_improvement'] < 0.05:
        analysis['overfitting_score'] += 2
        analysis['overfitting_reasons'].append("Train improves but val doesn't")
    
    if analysis['val_std_last10'] > 0.02:
        analysis['overfitting_score'] += 1
        analysis['overfitting_reasons'].append("High validation variance")
    
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
    
    if 'best_val_iou' in analysis:
        print(f"  Best Val IoU: {analysis['best_val_iou']:.4f}")
    if 'best_val_dice' in analysis:
        print(f"  Best Val Dice: {analysis['best_val_dice']:.4f}")
    
    print(f"\n📈 Improvement:")
    print(f"  Train: {analysis['train_improvement']:.4f}")
    print(f"  Val:   {analysis['val_improvement']:.4f}")
    
    print(f"\n🎯 Generalization Gap:")
    print(f"  Average: {analysis['avg_gap']:.4f}")
    print(f"  Final:   {analysis['final_gap']:.4f}")
    print(f"  Maximum: {analysis['max_gap']:.4f}")
    
    print(f"\n📊 Training Info:")
    print(f"  Total Epochs: {analysis['total_epochs']}")
    print(f"  Val Stability (std): {analysis['val_std_last10']:.4f}")
    
    print(f"\n⚠️  Overfitting Assessment:")
    print(f"  Status: {analysis['status']}")
    print(f"  Score:  {analysis['overfitting_score']}/5")
    
    if analysis['overfitting_reasons']:
        print(f"  Reasons: {', '.join(analysis['overfitting_reasons'])}")
    
    print(f"\n{'='*60}\n")


def create_comparison_table(experiments_data):
    """
    Create a pandas DataFrame comparison table
    
    Args:
        experiments_data: List of (name, analysis) tuples
    
    Returns:
        pandas DataFrame
    """
    data = []
    for name, analysis in experiments_data:
        if analysis:
            row = {
                'Name': name[:25] + '...' if len(name) > 25 else name,
                'Best Val Loss': f"{analysis['best_val_loss']:.4f}",
                'Best Epoch': analysis['best_val_epoch'],
                'Final Gap': f"{analysis['final_gap']:.4f}",
                'Epochs': analysis['total_epochs'],
                'Status': analysis['status']
            }
            if 'best_val_iou' in analysis:
                row['Best IoU'] = f"{analysis['best_val_iou']:.4f}"
            data.append(row)
    
    return pd.DataFrame(data)


def compare_experiments(filepaths, show_table=True):
    """
    Compare multiple experiments
    
    Args:
        filepaths: List of file paths
        show_table: Whether to show pandas table
    
    Returns:
        List of (name, history, analysis) tuples
    """
    experiments = []
    
    print(f"\n{'='*80}")
    print(f"{'EXPERIMENT COMPARISON':^80}")
    print(f"{'='*80}\n")
    
    for filepath in filepaths:
        if not Path(filepath).exists():
            print(f"⚠️  File not found: {filepath}")
            continue
        
        name, history, metadata = normalize_history(filepath)
        analysis = analyze_overfitting(history)
        experiments.append((name, history, analysis))
    
    if not experiments:
        print("❌ No valid experiments found")
        return []
    
    # Show pandas table
    if show_table:
        table_data = [(name, analysis) for name, _, analysis in experiments]
        df = create_comparison_table(table_data)
        print(df.to_string(index=False))
        print()
    
    # Find best experiment
    valid = [(name, analysis) for name, _, analysis in experiments if analysis]
    if valid:
        best = min(valid, key=lambda x: x[1]['best_val_loss'])
        print(f"\n🏆 Best Experiment: {best[0]}")
        print(f"   Best Val Loss: {best[1]['best_val_loss']:.4f}")
        print(f"   Status: {best[1]['status']}")
        if 'best_val_iou' in best[1]:
            print(f"   Best IoU: {best[1]['best_val_iou']:.4f}")
    
    print(f"\n{'='*80}\n")
    
    return experiments


def plot_comparison(experiments, save_path="experiment_comparison.png"):
    """
    Plot comparison of multiple experiments
    
    Args:
        experiments: List of (name, history, analysis) tuples
        save_path: Path to save plot
    """
    if not experiments:
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Validation Loss
    plt.subplot(2, 2, 1)
    for name, history, analysis in experiments:
        val_loss = history.get('val_loss', history.get('val_losses', []))
        epochs = history.get('epoch', list(range(1, len(val_loss) + 1)))
        label = name[:20] + '...' if len(name) > 20 else name
        if analysis:
            label += f" (Best: {analysis['best_val_loss']:.4f})"
        plt.plot(epochs, val_loss, label=label, linewidth=2)
    
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss
    plt.subplot(2, 2, 2)
    for name, history, analysis in experiments:
        train_loss = history.get('train_loss', history.get('train_losses', []))
        epochs = history.get('epoch', list(range(1, len(train_loss) + 1)))
        label = name[:20] + '...' if len(name) > 20 else name
        plt.plot(epochs, train_loss, label=label, linewidth=2)
    
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Generalization Gap
    plt.subplot(2, 2, 3)
    for name, history, analysis in experiments:
        train_loss = np.array(history.get('train_loss', history.get('train_losses', [])))
        val_loss = np.array(history.get('val_loss', history.get('val_losses', [])))
        if len(train_loss) > 0 and len(val_loss) > 0:
            gaps = val_loss - train_loss
            epochs = history.get('epoch', list(range(1, len(gaps) + 1)))
            label = name[:20] + '...' if len(name) > 20 else name
            plt.plot(epochs, gaps, label=label, linewidth=2)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.title('Generalization Gap (Val - Train)')
    plt.xlabel('Epoch')
    plt.ylabel('Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: IoU if available
    plt.subplot(2, 2, 4)
    has_iou = False
    for name, history, analysis in experiments:
        val_iou = history.get('val_iou', [])
        if val_iou:
            has_iou = True
            epochs = history.get('epoch', list(range(1, len(val_iou) + 1)))
            label = name[:20] + '...' if len(name) > 20 else name
            if analysis and 'best_val_iou' in analysis:
                label += f" (Best: {analysis['best_val_iou']:.4f})"
            plt.plot(epochs, val_iou, label=label, linewidth=2)
    
    if has_iou:
        plt.title('Validation IoU Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'IoU data not available', ha='center', va='center', fontsize=12)
        plt.title('Validation IoU (N/A)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {save_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Comprehensive experiment comparison tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single experiment
  python compare_experiments.py --history training_history.json
  
  # Compare multiple experiments
  python compare_experiments.py --compare exp1.json exp2.json exp3.json
  
  # Compare with plots
  python compare_experiments.py --compare exp1.json exp2.json --plot
  
  # Detailed analysis of each
  python compare_experiments.py --compare exp1.json exp2.json --detailed
        """
    )
    parser.add_argument('--history', type=str, default='training_history.json',
                        help='Path to training history file')
    parser.add_argument('--compare', nargs='+', type=str,
                        help='Compare multiple history/experiment files')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed analysis for each experiment')
    parser.add_argument('--output', type=str, default='experiment_comparison.png',
                        help='Output path for comparison plot')
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple experiments
        experiments = compare_experiments(args.compare)
        
        if args.detailed:
            for name, history, analysis in experiments:
                print_analysis(analysis, name=name)
        
        if args.plot and experiments:
            plot_comparison(experiments, save_path=args.output)
    
    else:
        # Single experiment analysis
        if not Path(args.history).exists():
            print(f"❌ File not found: {args.history}")
            return
        
        name, history, metadata = normalize_history(args.history)
        analysis = analyze_overfitting(history)
        print_analysis(analysis, name=name)
        
        if args.plot:
            plot_comparison([(name, history, analysis)], save_path=args.output)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # Default: analyze current training history
        history_path = Path("training_history.json")
        if history_path.exists():
            print("Analyzing current training history...")
            name, history, metadata = normalize_history(str(history_path))
            analysis = analyze_overfitting(history)
            print_analysis(analysis, name=name)
        else:
            print("No training_history.json found in current directory.")
            print("Usage: python compare_experiments.py --help")
    else:
        main()
