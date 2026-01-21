#!/usr/bin/env python3
"""
Training History Filter Tool

This script allows you to interactively filter training history,
keeping only the epochs you want.
"""

import json
import argparse
from pathlib import Path

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

def save_history(history, output_path):
    """Save filtered history to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved filtered history to: {output_path}")

def filter_history(history, epochs_to_keep):
    """Filter history to keep only specified epochs."""
    filtered = {
        'epoch': [],
        'train_losses': [],
        'val_losses': [],
        'learning rate(lr)': []
    }
    
    for i, epoch in enumerate(history['epoch']):
        if epoch in epochs_to_keep:
            filtered['epoch'].append(epoch)
            filtered['train_losses'].append(history['train_losses'][i])
            filtered['val_losses'].append(history['val_losses'][i])
            filtered['learning rate(lr)'].append(history['learning rate(lr)'][i])
    
    return filtered

def print_summary(original, filtered):
    """Print summary of filtering operation."""
    print("\n" + "="*60)
    print("FILTERING SUMMARY")
    print("="*60)
    print(f"Original epochs: {len(original['epoch'])}")
    print(f"Filtered epochs: {len(filtered['epoch'])}")
    print(f"Removed epochs:  {len(original['epoch']) - len(filtered['epoch'])}")
    print(f"Kept epochs:     {filtered['epoch']}")
    print("="*60 + "\n")

def interactive_filter(history):
    """Interactive mode to select epochs to keep."""
    print("\n" + "="*60)
    print("INTERACTIVE EPOCH SELECTION")
    print("="*60)
    print("\nAvailable epochs:", history['epoch'])
    print("\nOptions:")
    print("  1. Keep specific epochs (e.g., '1,5,10,21')")
    print("  2. Keep epoch range (e.g., '1-21')")
    print("  3. Keep best N epochs by validation loss")
    print("  4. Remove specific epochs")
    print("  5. Keep all epochs (no filtering)")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    epochs_to_keep = set()
    
    if choice == '1':
        epochs_str = input("Enter epoch numbers separated by commas: ").strip()
        try:
            epochs_to_keep = set(int(e.strip()) for e in epochs_str.split(',') if e.strip())
        except ValueError:
            print("❌ Error: Invalid input. Please enter numbers separated by commas.")
            epochs_to_keep = set(history['epoch'])
    
    elif choice == '2':
        range_str = input("Enter range (e.g., '1-21'): ").strip()
        try:
            start, end = map(int, range_str.split('-'))
            epochs_to_keep = set(range(start, end + 1))
        except ValueError:
            print("❌ Error: Invalid range format. Please use 'start-end' format.")
            epochs_to_keep = set(history['epoch'])
    
    elif choice == '3':
        try:
            n = int(input("Enter N (number of best epochs to keep): ").strip())
            # Sort epochs by validation loss and keep best N
            val_losses = history['val_losses']
            epoch_loss_pairs = list(zip(history['epoch'], val_losses))
            epoch_loss_pairs.sort(key=lambda x: x[1])
            epochs_to_keep = set(epoch for epoch, _ in epoch_loss_pairs[:n])
        except ValueError:
            print("❌ Error: Invalid number. Please enter a valid integer.")
            epochs_to_keep = set(history['epoch'])
    
    elif choice == '4':
        epochs_str = input("Enter epoch numbers to REMOVE separated by commas: ").strip()
        try:
            epochs_to_remove = set(int(e.strip()) for e in epochs_str.split(',') if e.strip())
            epochs_to_keep = set(history['epoch']) - epochs_to_remove
        except ValueError:
            print("❌ Error: Invalid input. Please enter numbers separated by commas.")
            epochs_to_keep = set(history['epoch'])
    
    elif choice == '5':
        epochs_to_keep = set(history['epoch'])
    
    else:
        print("Invalid choice. No filtering applied.")
        epochs_to_keep = set(history['epoch'])
    
    return sorted(epochs_to_keep)

def main():
    parser = argparse.ArgumentParser(
        description='Filter training history to keep only selected epochs'
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
        default='training_history_filtered.json',
        help='Output path for filtered history (default: training_history_filtered.json)'
    )
    parser.add_argument(
        '--keep-epochs',
        type=str,
        help='Comma-separated list of epochs to keep (e.g., "1,5,10,21")'
    )
    parser.add_argument(
        '--keep-range',
        type=str,
        help='Range of epochs to keep (e.g., "1-21")'
    )
    parser.add_argument(
        '--keep-best',
        type=int,
        help='Keep N best epochs by validation loss'
    )
    parser.add_argument(
        '--remove-epochs',
        type=str,
        help='Comma-separated list of epochs to remove'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode to select epochs'
    )
    
    args = parser.parse_args()
    
    # Load history
    print(f"Loading training history from: {args.history}")
    history = load_history(args.history)
    
    # Determine which epochs to keep
    epochs_to_keep = None
    
    if args.interactive:
        epochs_to_keep = interactive_filter(history)
    
    elif args.keep_epochs:
        try:
            epochs_to_keep = sorted(set(int(e.strip()) for e in args.keep_epochs.split(',') if e.strip()))
            print(f"Keeping epochs: {epochs_to_keep}")
        except ValueError:
            print("❌ Error: Invalid epoch numbers in --keep-epochs. Use comma-separated integers.")
            return
    
    elif args.keep_range:
        try:
            start, end = map(int, args.keep_range.split('-'))
            epochs_to_keep = list(range(start, end + 1))
            print(f"Keeping epochs from {start} to {end}")
        except ValueError:
            print("❌ Error: Invalid range format in --keep-range. Use 'start-end' format.")
            return
    
    elif args.keep_best:
        val_losses = history['val_losses']
        epoch_loss_pairs = list(zip(history['epoch'], val_losses))
        epoch_loss_pairs.sort(key=lambda x: x[1])
        epochs_to_keep = sorted([epoch for epoch, _ in epoch_loss_pairs[:args.keep_best]])
        print(f"Keeping {args.keep_best} best epochs by validation loss: {epochs_to_keep}")
    
    elif args.remove_epochs:
        try:
            epochs_to_remove = set(int(e.strip()) for e in args.remove_epochs.split(',') if e.strip())
            epochs_to_keep = sorted(set(history['epoch']) - epochs_to_remove)
            print(f"Removing epochs: {sorted(epochs_to_remove)}")
            print(f"Keeping epochs: {epochs_to_keep}")
        except ValueError:
            print("❌ Error: Invalid epoch numbers in --remove-epochs. Use comma-separated integers.")
            return
    
    else:
        print("No filtering option specified. Use --help for options.")
        return
    
    # Filter history
    filtered_history = filter_history(history, epochs_to_keep)
    
    # Print summary
    print_summary(history, filtered_history)
    
    # Save filtered history
    save_history(filtered_history, args.output)
    
    print("✅ Filtering complete!")

if __name__ == '__main__':
    main()
