"""
experiment_tracker.py - MLOps experiment tracking and logging
Supports MLflow for experiment tracking and model registry
"""

import os
import json
import yaml
from pathlib import Path
from datetime import datetime
import torch

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not available. Using file-based logging only.")


class ExperimentTracker:
    """
    Track experiments with MLflow or file-based logging
    """
    def __init__(self, experiment_name, tracking_uri="./mlruns", use_mlflow=True):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: URI for MLflow tracking server
            use_mlflow: Whether to use MLflow (if available)
        """
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        
        # Create local logging directory
        self.log_dir = Path("./experiments") / experiment_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if self.use_mlflow:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.run = mlflow.start_run()
            print(f"✓ MLflow tracking enabled: {experiment_name}")
            print(f"  Run ID: {self.run.info.run_id}")
        else:
            print(f"✓ File-based tracking: {self.log_dir}")
        
        self.metrics_history = []
    
    def log_params(self, params):
        """
        Log parameters
        
        Args:
            params: Dictionary of parameters to log
        """
        if self.use_mlflow:
            mlflow.log_params(params)
        
        # Always save to file
        with open(self.log_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)
    
    def log_config(self, config):
        """
        Log full configuration
        
        Args:
            config: Configuration dictionary
        """
        if self.use_mlflow:
            # Flatten config for MLflow
            flat_config = self._flatten_dict(config)
            mlflow.log_params(flat_config)
        
        # Save to file
        with open(self.log_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def log_metrics(self, metrics, step=None):
        """
        Log metrics for a step
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number (epoch number)
        """
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        
        # Add to history
        metrics_entry = {"step": step, **metrics}
        self.metrics_history.append(metrics_entry)
        
        # Save to file
        with open(self.log_dir / "metrics.json", "w") as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def log_model(self, model, model_name="model"):
        """
        Log model
        
        Args:
            model: PyTorch model
            model_name: Name for the model
        """
        model_path = self.log_dir / f"{model_name}.pt"
        
        if self.use_mlflow:
            mlflow.pytorch.log_model(model, model_name)
        
        # Save locally
        torch.save(model.state_dict(), model_path)
        print(f"  ✓ Model saved: {model_path}")
    
    def log_checkpoint(self, checkpoint, checkpoint_name="checkpoint"):
        """
        Log a full checkpoint (model + optimizer + epoch)
        
        Args:
            checkpoint: Dictionary containing model state, optimizer state, etc.
            checkpoint_name: Name for the checkpoint
        """
        checkpoint_path = self.log_dir / f"{checkpoint_name}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if self.use_mlflow:
            mlflow.log_artifact(str(checkpoint_path))
        
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    def log_artifact(self, artifact_path, artifact_name=None):
        """
        Log an artifact (file)
        
        Args:
            artifact_path: Path to the artifact
            artifact_name: Optional name for the artifact
        """
        if self.use_mlflow:
            mlflow.log_artifact(artifact_path, artifact_name)
        
        print(f"  ✓ Artifact logged: {artifact_path}")
    
    def end_run(self):
        """End the experiment run"""
        if self.use_mlflow:
            mlflow.end_run()
            print(f"✓ MLflow run ended")
    
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """
        Flatten nested dictionary for MLflow
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested items
            sep: Separator for nested keys
        
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class SimpleLogger:
    """
    Simple file-based logger for experiments without MLflow
    """
    def __init__(self, log_dir):
        """
        Initialize simple logger
        
        Args:
            log_dir: Directory for logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / "training.log"
        self.metrics_file = self.log_dir / "metrics.json"
        
        self.metrics_history = []
    
    def log(self, message):
        """
        Log a message
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, "a") as f:
            f.write(log_entry)
        
        print(log_entry.strip())
    
    def log_metrics(self, epoch, metrics):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        entry = {"epoch": epoch, **metrics}
        self.metrics_history.append(entry)
        
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.log(f"Epoch {epoch}: {metrics}")


def create_experiment_tracker(config, experiment_name=None):
    """
    Factory function to create experiment tracker
    
    Args:
        config: Configuration dictionary
        experiment_name: Name for the experiment (auto-generated if None)
    
    Returns:
        ExperimentTracker instance
    """
    if experiment_name is None:
        # Generate experiment name from config
        modality = config['data']['modality']
        encoder = config['model']['encoder_name']
        experiment_name = f"flood_detection_{modality}_{encoder}"
    
    tracker = ExperimentTracker(
        experiment_name=experiment_name,
        tracking_uri=config.get('mlflow', {}).get('tracking_uri', './mlruns'),
        use_mlflow=config.get('mlflow', {}).get('enabled', True)
    )
    
    return tracker


if __name__ == "__main__":
    # Test experiment tracker
    print("Testing experiment tracker...")
    
    # Create dummy config
    config = {
        'data': {'modality': 's1'},
        'model': {'encoder_name': 'resnet18'},
        'training': {'batch_size': 16, 'learning_rate': 0.0001}
    }
    
    # Create tracker
    tracker = ExperimentTracker("test_experiment", use_mlflow=False)
    
    # Log config
    tracker.log_config(config)
    
    # Log some metrics
    for epoch in range(3):
        metrics = {
            'train_loss': 0.5 - epoch * 0.1,
            'val_loss': 0.6 - epoch * 0.08,
            'iou': 0.5 + epoch * 0.1
        }
        tracker.log_metrics(metrics, step=epoch)
    
    # End run
    tracker.end_run()
    
    print("\n✓ Experiment tracker works correctly!")
    print(f"✓ Logs saved to: {tracker.log_dir}")
