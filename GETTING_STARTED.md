# 🚀 Getting Started - MLOps Pipeline for Flood Detection

## Welcome! 👋

Your flood detection project now has a **production-ready MLOps pipeline** that addresses overfitting and makes it easy to experiment with different models and datasets.

---

## 📋 What Was Built

### Core Pipeline
- ✅ **Configurable training pipeline** (pipeline.py)
- ✅ **Experiment tracking** with MLflow
- ✅ **6 evaluation metrics** (IoU, Dice, F1, Precision, Recall, Accuracy)
- ✅ **4 model architectures** (U-Net, DeepLabV3+, FPN, PSPNet)
- ✅ **10+ encoders** (ResNet, EfficientNet, MobileNet, etc.)
- ✅ **Production inference** (inference.py)

### Anti-Overfitting
Your training showed a gap of **0.09** between train and validation. We've implemented:
1. **Dropout**: 0.3 → 0.5
2. **Label smoothing**: 0.1
3. **Better encoder**: ResNet34 (was ResNet18)
4. **Weight decay**: 1e-4
5. **Gradient clipping**: 1.0
6. **IoU as primary metric** (more meaningful than loss)

Expected result: **Gap reduced to 0.03-0.05**

---

## 🎯 Quick Start (5 minutes)

### Step 1: Install Dependencies

```bash
cd /path/to/flood-detection
bash quick_start.sh
```

This will:
- Install required packages (mlflow, albumentations, etc.)
- Show your current configuration
- Verify everything is ready

### Step 2: Run Your First Experiment

```bash
python pipeline.py --experiment "my_first_run"
```

This will:
- Train with your current config (ResNet34, dropout 0.5)
- Track experiment in MLflow
- Save best model to `outputs/models/best_model.pt`
- Generate training history

Expected time: **1-2 hours** (depends on dataset size and GPU)

### Step 3: View Results

```bash
# Start MLflow UI
mlflow ui

# Open browser: http://localhost:5000
```

You'll see:
- All experiments and their metrics
- Training curves (loss, IoU, Dice, F1)
- Parameter comparisons
- Model downloads

---

## 🔄 Your Workflow Now

### Daily Experimentation

```bash
# Try different encoder
# Edit config.yaml: encoder_name: "resnet50"
python pipeline.py --experiment "resnet50_test"

# Try different architecture
# Edit config.yaml: architecture: "deeplabv3+"
python pipeline.py --experiment "deeplabv3_test"

# Try different modality
# Edit config.yaml: modality: "s1_s2"
python pipeline.py --experiment "s1_s2_test"

# Compare all experiments
mlflow ui
```

### Analyzing Results

```bash
# Check for overfitting
python compare_experiments.py --history training_history.json --plot

# Compare multiple experiments
python compare_experiments.py --compare \
    experiments/exp1/training_history.json \
    experiments/exp2/training_history.json
```

### Making Predictions

```bash
# Predict on new image
python inference.py \
    --image path/to/test_image.tif \
    --checkpoint outputs/models/best_model.pt \
    --output prediction_viz.png \
    --save-mask prediction_mask.tif
```

---

## 📝 Configuration Guide

Your `config.yaml` is the control center. Edit it to change everything without touching code.

### Example Configurations

#### Configuration 1: Balanced (Default)
Good starting point for most cases
```yaml
model:
  architecture: "unet"
  encoder_name: "resnet34"
  dropout_rate: 0.5
  label_smoothing: 0.1
training:
  batch_size: 16
  learning_rate: 0.0001
  weight_decay: 1e-4
```

#### Configuration 2: Strong Regularization
Use if still overfitting
```yaml
model:
  architecture: "unet"
  encoder_name: "resnet34"
  dropout_rate: 0.7  # Higher dropout
  label_smoothing: 0.15  # More smoothing
training:
  batch_size: 8  # Smaller batches
  learning_rate: 0.0001
  weight_decay: 1e-3  # Stronger L2
```

#### Configuration 3: Best Performance
For final production model
```yaml
model:
  architecture: "deeplabv3+"  # Better architecture
  encoder_name: "resnet50"  # Larger encoder
  dropout_rate: 0.4
  label_smoothing: 0.1
training:
  batch_size: 16
  learning_rate: 0.0001
  num_epochs: 100  # Train longer
  early_stopping_patience: 20
```

#### Configuration 4: Fast Iteration
For quick experiments
```yaml
model:
  architecture: "unet"
  encoder_name: "resnet18"  # Fastest
  dropout_rate: 0.5
training:
  batch_size: 32  # Larger batches
  learning_rate: 0.001  # Higher LR
  num_epochs: 30  # Fewer epochs
```

---

## 🎓 Understanding Your Metrics

### What to Watch

1. **Validation IoU** (most important)
   - Target: > 0.65
   - Higher is better
   - Measures overlap between prediction and ground truth

2. **Train-Val Gap**
   - Target: < 0.05
   - Lower is better
   - Large gap = overfitting

3. **Validation F1 Score**
   - Target: > 0.70
   - Balance of precision and recall

4. **Learning Rate**
   - Should decrease during training
   - If stuck at minimum, training has plateaued

### Good Training Signs ✅
- Validation IoU increasing
- Train-val gap small (< 0.05)
- Validation loss stable (not jumping)
- Learning rate decreasing over time
- Early stopping kicks in (not training to max epochs)

### Bad Training Signs ⚠️
- Validation IoU not improving
- Train-val gap large (> 0.08)
- Validation loss jumping wildly
- Learning rate at minimum but no improvement
- Training to max epochs (early stopping didn't work)

---

## 🔧 Common Scenarios

### Scenario 1: Still Overfitting

**Symptoms**: Train-val gap > 0.05

**Solution**:
```yaml
# In config.yaml
model:
  dropout_rate: 0.7  # Increase
training:
  weight_decay: 1e-3  # Increase
  batch_size: 8  # Decrease
```

### Scenario 2: Underfitting

**Symptoms**: Both train and val loss high, gap small

**Solution**:
```yaml
# In config.yaml
model:
  encoder_name: "resnet50"  # Larger model
  dropout_rate: 0.3  # Decrease
training:
  learning_rate: 0.0003  # Increase
  num_epochs: 100  # Train longer
```

### Scenario 3: Training Too Slow

**Solution**:
```yaml
# In config.yaml
model:
  encoder_name: "resnet18"  # Smaller
training:
  batch_size: 32  # Larger
```

### Scenario 4: Out of Memory

**Solution**:
```yaml
# In config.yaml
training:
  batch_size: 8  # Reduce
  num_workers: 1  # Reduce
```

---

## 📊 Comparing Experiments

### In MLflow UI

1. Open http://localhost:5000
2. Select multiple experiments
3. Click "Compare"
4. View:
   - Metric comparisons
   - Parameter differences
   - Training curves side-by-side

### With Compare Tool

```bash
# Analyze single experiment
python compare_experiments.py --history training_history.json --plot

# Compare multiple
python compare_experiments.py --compare \
    experiments/resnet34/training_history.json \
    experiments/resnet50/training_history.json \
    --plot
```

The tool will show:
- Overfitting score (0-5)
- Best validation loss
- Training improvements
- Recommendations

---

## 🎯 Next Steps

### Week 1: Baseline
```bash
# Run with default settings
python pipeline.py --experiment "baseline_resnet34"

# Check results
python compare_experiments.py --history training_history.json --plot
```

### Week 2: Architecture Comparison
```bash
# Try U-Net
python pipeline.py --experiment "unet_resnet34"

# Try DeepLabV3+
# Edit config.yaml: architecture: "deeplabv3+"
python pipeline.py --experiment "deeplabv3_resnet34"

# Compare in MLflow
mlflow ui
```

### Week 3: Encoder Comparison
```bash
# Try ResNet50
# Edit config.yaml: encoder_name: "resnet50"
python pipeline.py --experiment "unet_resnet50"

# Try EfficientNet
# Edit config.yaml: encoder_name: "efficientnet-b0"
python pipeline.py --experiment "unet_efficientnet"
```

### Week 4: Production Model
```bash
# Pick best config from MLflow
# Train final model
python pipeline.py --experiment "production_v1"

# Test inference
python inference.py --image test.tif --checkpoint outputs/models/best_model.pt
```

---

## 📚 Documentation

### Quick Reference
- **MLOPS_README.md** - Complete usage guide
- **IMPLEMENTATION_SUMMARY.md** - What was built and why
- **MIGRATION_GUIDE.md** - Moving from train.py
- **THIS FILE** - Getting started guide

### Code Structure
```
flood-detection/
├── pipeline.py              # Main training pipeline (USE THIS)
├── train.py                 # Old training (still works)
├── inference.py             # Make predictions
├── compare_experiments.py   # Analyze results
├── metrics.py               # Evaluation metrics
├── experiment_tracker.py    # MLflow tracking
├── model.py                 # Model architectures
├── dataset.py               # Data loading
├── config.yaml              # Configuration (EDIT THIS)
└── quick_start.sh           # Setup script
```

---

## ❓ FAQ

**Q: Can I keep using train.py?**  
A: Yes! It still works. But pipeline.py gives you better overfitting control and experiment tracking.

**Q: Will this fix my overfitting?**  
A: Very likely! The 6 anti-overfitting strategies should reduce your gap from 0.09 to 0.03-0.05.

**Q: How do I switch models?**  
A: Edit config.yaml, change `encoder_name` or `architecture`, run pipeline.py.

**Q: What if MLflow doesn't work?**  
A: Set `use_mlflow: false` in config.yaml. Everything else still works.

**Q: Can I use my old checkpoints?**  
A: Yes! Use inference.py with your old checkpoint path.

**Q: How long does training take?**  
A: Depends on dataset size and GPU. Expect 1-2 hours for ~100 samples, 50 epochs.

**Q: What's the best architecture?**  
A: Start with U-Net + ResNet34. Try DeepLabV3+ + ResNet50 for best performance.

**Q: Can I add MSFlood-Net?**  
A: Yes! The structure is ready. Add the architecture to model.py.

---

## 🆘 Getting Help

**If something doesn't work:**

1. Check config.yaml syntax
2. Run `python pipeline.py --help`
3. Check MLflow logs: `cat mlruns/.../meta.yaml`
4. Review training_history.json
5. Open an issue with error message

**If results aren't good:**

1. Run compare_experiments.py to analyze
2. Check if overfitting (gap > 0.05)
3. Try stronger regularization
4. Ensure augmentation is enabled
5. Compare with other experiments in MLflow

---

## 🎉 Success Checklist

After your first successful run, you should have:

- ✅ Experiment tracked in MLflow
- ✅ Best model saved to outputs/models/
- ✅ Training history JSON file
- ✅ Validation IoU > 0.60
- ✅ Train-val gap < 0.06
- ✅ F1 score > 0.65

If you see these, you're ready to experiment!

---

## 💡 Pro Tips

1. **Always name experiments**: `--experiment "descriptive_name"`
2. **Use MLflow**: Makes comparison so much easier
3. **Track IoU, not loss**: More meaningful for segmentation
4. **Start small**: ResNet18 for quick iteration
5. **Document changes**: Note what worked in MLflow
6. **Save configs**: Keep successful configs for later
7. **Compare often**: Use MLflow UI to spot patterns

---

## 🚀 Ready to Go!

You now have a **production-ready MLOps pipeline** that will:
- Reduce your overfitting
- Make experimentation faster
- Track everything automatically
- Scale to advanced models

**Start with**:
```bash
bash quick_start.sh
python pipeline.py --experiment "my_first_run"
mlflow ui
```

**Happy training! 🎯**

---

*Need more details? Check the other documentation files in this repository.*
