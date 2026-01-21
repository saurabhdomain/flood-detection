# Flood Detection using Sentinel-1 and Sentinel-2 Data

Semantic segmentation for flood detection using U-Net architecture with Sentinel SAR and optical imagery.

## Dataset
STURM-Flood dataset from Zenodo

## Setup
```bash
pip install -r requirements.txt
```

## Run the code in this order

1. setup.py
2. dataset_download.py
3. explore_data.py
4. dataset.py
5. model.py
6. train.py

## Training History Analysis Tools

After training, you can analyze and manage your training history:

### Visualize Training History
```bash
# Generate comprehensive analysis with visualizations
python visualize_history.py

# View the generated plot: training_history_analysis.png
```

This will:
- Show loss curves with key epochs highlighted (best validation, overfitting start, etc.)
- Provide epoch-by-epoch breakdown
- Recommend which epochs to keep or discard

### Filter Training History
```bash
# Keep only the best epochs by validation loss
python filter_history.py --keep-best 5

# Keep epochs in a specific range
python filter_history.py --keep-range "1-21"

# Interactive mode to select epochs
python filter_history.py --interactive
```

📖 **See [HISTORY_TOOLS_README.md](HISTORY_TOOLS_README.md) for detailed documentation and examples.**
