"""
save_experiment.py - Save experiment results to a shareable format
This allows comparing experiments across different systems
"""

import json
import platform
import socket
import hashlib
from datetime import datetime
from pathlib import Path
import argparse


def get_system_info():
    """Get information about the current system"""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version()
    }


def generate_experiment_id(name, timestamp):
    """Generate a unique experiment ID"""
    hash_input = f"{name}_{timestamp}_{socket.gethostname()}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:12]


def save_experiment(
    experiment_name,
    description="",
    notes="",
    output_dir="./experiment_results"
):
    """
    Save current experiment results to a shareable JSON file
    
    Args:
        experiment_name: Name/identifier for this experiment
        description: Brief description of what was tested
        notes: Any additional notes
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    experiment_id = generate_experiment_id(experiment_name, timestamp)
    
    # Load training history
    history_path = Path("outputs/models/training_history.json")
    training_history = {}
    if history_path.exists():
        with open(history_path, 'r') as f:
            training_history = json.load(f)
    
    # Load config
    config_path = Path("config.yaml")
    config = {}
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Calculate summary metrics
    train_losses = training_history.get('train_losses', [])
    val_losses = training_history.get('val_losses', [])
    
    summary = {}
    if train_losses and val_losses:
        import numpy as np
        summary = {
            "total_epochs": len(train_losses),
            "best_val_loss": float(min(val_losses)),
            "best_val_epoch": int(np.argmin(val_losses) + 1),
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(val_losses[-1]),
            "train_improvement": float(train_losses[0] - train_losses[-1]),
            "val_improvement": float(val_losses[0] - val_losses[-1]),
            "avg_generalization_gap": float(np.mean(np.array(val_losses) - np.array(train_losses)))
        }
    
    # Build experiment record
    experiment = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "description": description,
        "notes": notes,
        "timestamp": timestamp,
        "system_info": get_system_info(),
        "config": config,
        "summary": summary,
        "training_history": training_history
    }
    
    # Save to file
    filename = f"{experiment_name}_{experiment_id}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(experiment, f, indent=2)
    
    print(f"\n{'='*60}")
    print("📁 EXPERIMENT SAVED")
    print(f"{'='*60}")
    print(f"  Experiment ID:   {experiment_id}")
    print(f"  Name:            {experiment_name}")
    print(f"  System:          {get_system_info()['hostname']}")
    print(f"  Timestamp:       {timestamp}")
    print(f"  File:            {filepath}")
    print(f"{'='*60}")
    
    if summary:
        print(f"\n📊 SUMMARY METRICS:")
        print(f"  Total Epochs:      {summary['total_epochs']}")
        print(f"  Best Val Loss:     {summary['best_val_loss']:.4f} (Epoch {summary['best_val_epoch']})")
        print(f"  Final Train Loss:  {summary['final_train_loss']:.4f}")
        print(f"  Final Val Loss:    {summary['final_val_loss']:.4f}")
        print(f"  Avg Gap:           {summary['avg_generalization_gap']:.4f}")
    
    print(f"\n✅ Share this file to compare with other experiments!")
    print(f"   Use: python compare_saved_experiments.py --files <file1> <file2>")
    
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save experiment results")
    parser.add_argument("--name", "-n", type=str, default="experiment", help="Experiment name")
    parser.add_argument("--desc", "-d", type=str, default="", help="Description")
    parser.add_argument("--notes", type=str, default="", help="Additional notes")
    parser.add_argument("--output", "-o", type=str, default="./experiment_results", help="Output directory")
    
    args = parser.parse_args()
    
    save_experiment(
        experiment_name=args.name,
        description=args.desc,
        notes=args.notes,
        output_dir=args.output
    )
