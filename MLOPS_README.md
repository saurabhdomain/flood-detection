# Flood Detection MLOps Pipeline

A configurable, production-ready MLOps pipeline for flood detection using deep learning with Sentinel-1 SAR and Sentinel-2 optical imagery.

## 🎯 Features

### MLOps Capabilities
- ✅ **Experiment Tracking**: Integrated MLflow for tracking experiments, parameters, and metrics
- ✅ **Model Registry**: Version and manage trained models
- ✅ **Configurable Pipeline**: Easy switching between models, encoders, and datasets
- ✅ **Comprehensive Metrics**: IoU, Dice, F1, Precision, Recall, and Pixel Accuracy
- ✅ **Advanced Regularization**: Dropout, label smoothing, weight decay to combat overfitting
- ✅ **Data Augmentation**: Extensive augmentation pipeline for better generalization

### Model Flexibility
- **Architectures**: U-Net, DeepLabV3+, FPN, PSPNet
- **Encoders**: ResNet (18/34/50/101), EfficientNet (b0-b7), MobileNetV2, and more
- **Data Modalities**: Sentinel-1 (SAR), Sentinel-2 (Optical), or Both Combined
- **Future Ready**: Structure supports advanced models like MSFlood-Net, FloodSformer

## 📁 Project Structure

```
flood-detection/
├── config.yaml              # Main configuration file
├── pipeline.py              # Main MLOps pipeline orchestrator
├── model.py                 # Model architectures
├── dataset.py               # Dataset loading and augmentation
├── metrics.py               # Evaluation metrics
├── experiment_tracker.py    # Experiment tracking utilities
├── train.py                 # Legacy training script
├── requirements.txt         # Python dependencies
├── outputs/                 # Training outputs
│   ├── models/             # Saved models
│   ├── logs/               # Training logs
│   └── plots/              # Visualizations
├── experiments/            # Experiment tracking data
└── mlruns/                 # MLflow tracking data
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/saurabhdomain/flood-detection.git
cd flood-detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Your Experiment

Edit `config.yaml` to customize your training:

```yaml
# Choose data modality
data:
  modality: "s1"  # Options: s1, s2, s1_s2
  augmentation: True

# Choose model architecture
model:
  architecture: "unet"  # Options: unet, deeplabv3+, fpn, pspnet
  encoder_name: "resnet34"  # Try: resnet18, resnet50, efficientnet-b0
  dropout_rate: 0.5
  label_smoothing: 0.1

# Training configuration
training:
  batch_size: 16
  learning_rate: 0.0001
  weight_decay: 1e-4
  early_stopping_patience: 15
```

### 3. Run Training Pipeline

```bash
# Basic training
python pipeline.py

# Custom experiment name
python pipeline.py --experiment "my_experiment_v1"

# Custom config file
python pipeline.py --config my_config.yaml
```

### 4. Monitor Experiments

```bash
# Start MLflow UI
mlflow ui

# Open browser at http://localhost:5000
```

## 🔧 Configuration Guide

### Data Configuration

| Parameter | Options | Description |
|-----------|---------|-------------|
| `modality` | s1, s2, s1_s2 | Input data type |
| `mask_source` | s1, s2 | Which mask to use |
| `augmentation` | true/false | Enable data augmentation |

### Model Configuration

| Parameter | Options | Description |
|-----------|---------|-------------|
| `architecture` | unet, deeplabv3+, fpn, pspnet | Model architecture |
| `encoder_name` | resnet18, resnet34, resnet50, efficientnet-b0, etc. | Encoder backbone |
| `dropout_rate` | 0.0-0.8 | Dropout for regularization |
| `label_smoothing` | 0.0-0.3 | Label smoothing factor |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Training batch size |
| `learning_rate` | 0.0001 | Initial learning rate |
| `weight_decay` | 1e-4 | L2 regularization |
| `early_stopping_patience` | 15 | Epochs before stopping |

## 📊 Addressing Overfitting

The pipeline includes multiple strategies to combat overfitting:

1. **Regularization**:
   - Increased dropout (0.5 vs 0.3)
   - Weight decay (L2 regularization)
   - Label smoothing

2. **Data Augmentation**:
   - Spatial transforms (flip, rotate, affine)
   - Elastic deformations
   - Coarse dropout (cutout)
   - Brightness/contrast adjustments

3. **Training Strategies**:
   - Early stopping with patience
   - Learning rate scheduling
   - Gradient clipping

4. **Monitoring**:
   - Track train/val gap
   - Multiple evaluation metrics
   - Experiment comparison

## 📈 Metrics Tracked

- **Loss**: Training and validation loss
- **IoU**: Intersection over Union
- **Dice**: Dice coefficient (F1 for segmentation)
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Pixel Accuracy**: Overall pixel classification accuracy

## 🔄 Comparing Experiments

```python
# Try different configurations
python pipeline.py --experiment "resnet18_exp"
# Edit config.yaml: encoder_name: "resnet34"
python pipeline.py --experiment "resnet34_exp"
# Edit config.yaml: architecture: "deeplabv3+"
python pipeline.py --experiment "deeplabv3_exp"

# Compare in MLflow UI
mlflow ui
```

## 🎓 Training History Analysis

The pipeline saves detailed training history including:
- Loss curves (train/val)
- Metric evolution (IoU, Dice, F1)
- Learning rate schedule
- Model checkpoints

Access via `outputs/models/training_history.json`

## 🔮 Future Enhancements

### Phase 1 (Current): MLOps Foundation ✅
- [x] Modular pipeline structure
- [x] Experiment tracking
- [x] Configurable models and datasets
- [x] Comprehensive metrics
- [x] Overfitting mitigation

### Phase 2: Advanced Models (Planned)
- [ ] MSFlood-Net integration
- [ ] FloodSformer architecture
- [ ] Transformer-based encoders
- [ ] Ensemble methods

### Phase 3: Production Features (Planned)
- [ ] Real-time inference API
- [ ] Model serving (TorchServe/ONNX)
- [ ] "What-if" scenario analysis
- [ ] Damage assessment integration
- [ ] Interactive flood mapping dashboard

### Phase 4: Extended Datasets (Planned)
- [ ] Additional flood datasets
- [ ] Multi-region support
- [ ] Temporal analysis
- [ ] Climate data integration

## 🐛 Troubleshooting

### Out of Memory
```yaml
# Reduce batch size
training:
  batch_size: 8
```

### Training Too Slow
```yaml
# Use smaller encoder
model:
  encoder_name: "resnet18"  # Instead of resnet50
```

### Still Overfitting
```yaml
# Increase regularization
model:
  dropout_rate: 0.7  # Higher dropout
training:
  weight_decay: 1e-3  # Stronger L2
```

## 📚 References

- **Segmentation Models PyTorch**: https://github.com/qubvel/segmentation_models.pytorch
- **MLflow**: https://mlflow.org/
- **STURM-Flood Dataset**: Zenodo

## 📄 License

[Your License Here]

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📧 Contact

For questions or support, please open an issue on GitHub.
