#!/bin/bash

# Quick Start Script for Flood Detection MLOps Pipeline
# This script helps you get started with the pipeline quickly

set -e

echo "======================================"
echo "Flood Detection MLOps Pipeline Setup"
echo "======================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "✓ Python version: $python_version"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Installation complete!"
echo ""

# Check configuration
echo "======================================"
echo "Configuration Summary"
echo "======================================"
echo ""

# Parse config.yaml (simple grep-based parsing)
modality=$(grep "modality:" config.yaml | awk '{print $2}' | tr -d '"')
architecture=$(grep "architecture:" config.yaml | awk '{print $2}' | tr -d '"')
encoder=$(grep "encoder_name:" config.yaml | awk '{print $2}' | tr -d '"')
batch_size=$(grep "batch_size:" config.yaml | awk '{print $2}')
learning_rate=$(grep "learning_rate:" config.yaml | awk '{print $2}')

echo "Data Configuration:"
echo "  Modality: $modality"
echo ""
echo "Model Configuration:"
echo "  Architecture: $architecture"
echo "  Encoder: $encoder"
echo ""
echo "Training Configuration:"
echo "  Batch Size: $batch_size"
echo "  Learning Rate: $learning_rate"
echo ""

echo "======================================"
echo "Ready to Start Training!"
echo "======================================"
echo ""
echo "Run the pipeline with:"
echo "  python pipeline.py"
echo ""
echo "Or with custom experiment name:"
echo "  python pipeline.py --experiment my_experiment"
echo ""
echo "To monitor experiments:"
echo "  mlflow ui"
echo "  Then open: http://localhost:5000"
echo ""
echo "For more information, see MLOPS_README.md"
echo ""
