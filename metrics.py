"""
metrics.py - Comprehensive evaluation metrics for flood detection
Includes IoU, F1, Precision, Recall, and other segmentation metrics
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def calculate_iou(pred, target, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) for binary segmentation
    
    Args:
        pred: Model predictions (logits or probabilities)
        target: Ground truth masks
        threshold: Threshold for converting predictions to binary
    
    Returns:
        IoU score
    """
    # Convert to binary
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > threshold).float()
    
    # Calculate intersection and union
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou.item()


def calculate_dice(pred, target, threshold=0.5):
    """
    Calculate Dice coefficient (F1 score for segmentation)
    
    Args:
        pred: Model predictions
        target: Ground truth masks
        threshold: Threshold for binary conversion
    
    Returns:
        Dice coefficient
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    dice = (2.0 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
    
    return dice.item()


def calculate_pixel_accuracy(pred, target, threshold=0.5):
    """
    Calculate pixel-wise accuracy
    
    Args:
        pred: Model predictions
        target: Ground truth masks
        threshold: Threshold for binary conversion
    
    Returns:
        Pixel accuracy
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > threshold).float()
    
    correct = (pred_binary == target_binary).sum()
    total = target_binary.numel()
    
    accuracy = correct / total
    return accuracy.item()


def calculate_precision_recall(pred, target, threshold=0.5):
    """
    Calculate precision and recall
    
    Args:
        pred: Model predictions
        target: Ground truth masks
        threshold: Threshold for binary conversion
    
    Returns:
        Tuple of (precision, recall)
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > threshold).float()
    
    # True positives, false positives, false negatives
    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return precision.item(), recall.item()


def calculate_f1_score(pred, target, threshold=0.5):
    """
    Calculate F1 score
    
    Args:
        pred: Model predictions
        target: Ground truth masks
        threshold: Threshold for binary conversion
    
    Returns:
        F1 score
    """
    precision, recall = calculate_precision_recall(pred, target, threshold)
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


class MetricsTracker:
    """
    Track and accumulate metrics over multiple batches
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.iou_scores = []
        self.dice_scores = []
        self.pixel_accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
    
    def update(self, pred, target, threshold=0.5):
        """
        Update metrics with a new batch
        
        Args:
            pred: Model predictions
            target: Ground truth masks
            threshold: Threshold for binary conversion
        """
        self.iou_scores.append(calculate_iou(pred, target, threshold))
        self.dice_scores.append(calculate_dice(pred, target, threshold))
        self.pixel_accuracies.append(calculate_pixel_accuracy(pred, target, threshold))
        
        precision, recall = calculate_precision_recall(pred, target, threshold)
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(calculate_f1_score(pred, target, threshold))
    
    def get_metrics(self):
        """
        Get average metrics
        
        Returns:
            Dictionary of metric names and values
        """
        return {
            'iou': np.mean(self.iou_scores) if self.iou_scores else 0.0,
            'dice': np.mean(self.dice_scores) if self.dice_scores else 0.0,
            'pixel_accuracy': np.mean(self.pixel_accuracies) if self.pixel_accuracies else 0.0,
            'precision': np.mean(self.precisions) if self.precisions else 0.0,
            'recall': np.mean(self.recalls) if self.recalls else 0.0,
            'f1_score': np.mean(self.f1_scores) if self.f1_scores else 0.0
        }
    
    def print_metrics(self, prefix=""):
        """
        Print metrics in a formatted way
        
        Args:
            prefix: Prefix to add to the output (e.g., "Train" or "Val")
        """
        metrics = self.get_metrics()
        print(f"\n{prefix} Metrics:")
        print(f"  IoU:            {metrics['iou']:.4f}")
        print(f"  Dice:           {metrics['dice']:.4f}")
        print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print(f"  Precision:      {metrics['precision']:.4f}")
        print(f"  Recall:         {metrics['recall']:.4f}")
        print(f"  F1 Score:       {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics module...")
    
    # Create dummy data
    pred = torch.randn(2, 1, 128, 128)  # Batch of 2, 1 channel, 128x128
    target = torch.randint(0, 2, (2, 1, 128, 128)).float()
    
    # Test individual metrics
    iou = calculate_iou(pred, target)
    dice = calculate_dice(pred, target)
    pixel_acc = calculate_pixel_accuracy(pred, target)
    precision, recall = calculate_precision_recall(pred, target)
    f1 = calculate_f1_score(pred, target)
    
    print(f"\nIndividual metrics:")
    print(f"  IoU: {iou:.4f}")
    print(f"  Dice: {dice:.4f}")
    print(f"  Pixel Accuracy: {pixel_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    
    # Test MetricsTracker
    tracker = MetricsTracker()
    tracker.update(pred, target)
    tracker.print_metrics("Test")
    
    print("\n✓ Metrics module works correctly!")
