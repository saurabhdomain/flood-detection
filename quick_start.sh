#!/bin/bash

# Quick Start Script for Flood Detection MLOps Pipeline
# This script handles the complete workflow from setup to experiment comparison

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

VENV_NAME=".geoai"

print_header() {
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Parse command line arguments
SKIP_VENV=false
SKIP_INSTALL=false
SKIP_EXPLORE=false
SKIP_TRAIN=false
EXPERIMENT_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-venv)
            SKIP_VENV=true
            shift
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --skip-explore)
            SKIP_EXPLORE=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --experiment)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-venv      Skip virtual environment creation"
            echo "  --skip-install   Skip dependency installation"
            echo "  --skip-explore   Skip dataset exploration"
            echo "  --skip-train     Skip training (only setup)"
            echo "  --experiment     Set custom experiment name"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header "🚀 Flood Detection MLOps Pipeline"

# ============================================
# STEP 1: Create Virtual Environment
# ============================================
if [ "$SKIP_VENV" = false ]; then
    print_header "Step 1: Creating Virtual Environment"
    
    if [ -d "$VENV_NAME" ]; then
        print_warning "Virtual environment '$VENV_NAME' already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_NAME"
            python3 -m venv "$VENV_NAME"
            print_step "Recreated virtual environment: $VENV_NAME"
        else
            print_step "Using existing virtual environment: $VENV_NAME"
        fi
    else
        python3 -m venv "$VENV_NAME"
        print_step "Created virtual environment: $VENV_NAME"
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    print_step "Activated virtual environment"
    
    # Upgrade pip
    pip install --upgrade pip > /dev/null 2>&1
    print_step "Upgraded pip"
else
    print_warning "Skipping virtual environment creation"
    if [ -d "$VENV_NAME" ]; then
        source "$VENV_NAME/bin/activate"
        print_step "Activated existing virtual environment"
    fi
fi

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
print_step "Python version: $python_version"

# ============================================
# STEP 2: Install Dependencies
# ============================================
if [ "$SKIP_INSTALL" = false ]; then
    print_header "Step 2: Installing Dependencies"
    
    echo "📦 Installing requirements.txt..."
    pip install -r requirements.txt
    print_step "Installed requirements.txt"
    
    # Run setup.py if it exists
    if [ -f "setup.py" ]; then
        echo ""
        echo "📦 Installing package via setup.py..."
        pip install -e .
        print_step "Installed package in editable mode"
    fi
else
    print_warning "Skipping dependency installation"
fi

# ============================================
# STEP 3: Display Configuration
# ============================================
print_header "Step 3: Configuration Summary"

if [ -f "config.yaml" ]; then
    modality=$(grep "modality:" config.yaml | head -1 | awk '{print $2}' | tr -d '"')
    architecture=$(grep "architecture:" config.yaml | head -1 | awk '{print $2}' | tr -d '"')
    encoder=$(grep "encoder_name:" config.yaml | head -1 | awk '{print $2}' | tr -d '"')
    batch_size=$(grep "batch_size:" config.yaml | head -1 | awk '{print $2}')
    learning_rate=$(grep "learning_rate:" config.yaml | head -1 | awk '{print $2}')
    num_epochs=$(grep "num_epochs:" config.yaml | head -1 | awk '{print $2}')
    
    echo "📊 Data Configuration:"
    echo "   Modality: $modality"
    echo ""
    echo "🏗️  Model Configuration:"
    echo "   Architecture: $architecture"
    echo "   Encoder: $encoder"
    echo ""
    echo "🎯 Training Configuration:"
    echo "   Batch Size: $batch_size"
    echo "   Learning Rate: $learning_rate"
    echo "   Epochs: $num_epochs"
else
    print_error "config.yaml not found!"
    exit 1
fi

# ============================================
# STEP 4: Explore Dataset
# ============================================
if [ "$SKIP_EXPLORE" = false ]; then
    print_header "Step 4: Exploring Dataset"
    
    if [ -f "explore_data.py" ]; then
        echo "🔍 Running dataset exploration..."
        python explore_data.py
        print_step "Dataset exploration complete"
    else
        print_warning "explore_data.py not found, skipping exploration"
    fi
else
    print_warning "Skipping dataset exploration"
fi

# ============================================
# STEP 5: Train Model
# ============================================
if [ "$SKIP_TRAIN" = false ]; then
    print_header "Step 5: Training Model"
    
    echo "🎯 Starting training..."
    echo ""
    
    if [ -n "$EXPERIMENT_NAME" ]; then
        echo "Experiment name: $EXPERIMENT_NAME"
        python pipeline.py --experiment "$EXPERIMENT_NAME"
    else
        python pipeline.py
    fi
    
    print_step "Training complete!"
    
    # ============================================
    # STEP 6: Save Experiment
    # ============================================
    print_header "Step 6: Saving Experiment"
    
    if [ -f "save_experiment.py" ]; then
        echo "💾 Saving experiment results..."
        python save_experiment.py
        print_step "Experiment saved"
    else
        print_warning "save_experiment.py not found, skipping"
    fi
    
    # ============================================
    # STEP 7: Compare Experiments
    # ============================================
    print_header "Step 7: Comparing Experiments"
    
    if [ -f "compare_experiments.py" ]; then
        echo "📈 Analyzing and comparing experiments..."
        python compare_experiments.py --plot
        print_step "Experiment analysis complete"
        print_step "Comparison plot saved to: experiment_comparison.png"
    else
        print_warning "compare_experiments.py not found, skipping"
    fi
else
    print_warning "Skipping training, saving, and comparison"
fi

# ============================================
# SUMMARY
# ============================================
print_header "🎉 Pipeline Complete!"

echo "Summary of actions:"
echo "  ✓ Virtual environment: $VENV_NAME"
echo "  ✓ Dependencies installed"
if [ "$SKIP_EXPLORE" = false ]; then
    echo "  ✓ Dataset explored"
fi
if [ "$SKIP_TRAIN" = false ]; then
    echo "  ✓ Model trained"
    echo "  ✓ Experiment saved"
    echo "  ✓ Experiments compared"
fi

echo ""
echo "Next steps:"
echo "  • Activate venv:    source $VENV_NAME/bin/activate"
echo "  • View MLflow UI:   mlflow ui"
echo "  • Run inference:    python inference.py --image <path>"
echo "  • Compare runs:     python compare_saved_experiments.py"
echo ""
echo "For more information, see MLOPS_README.md"
echo ""
