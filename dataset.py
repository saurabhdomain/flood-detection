"""
dataset.py - Load S1, S2, or both with appropriate masks
Updated to handle S1 masks and S2 masks separately
"""

import os
from pathlib import Path
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
import torch
import yaml

#-----------------------------------
# Load Configuration
#-----------------------------------

def load_config(config_path = 'config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

class FloodDataset(Dataset):
    """ 
    Class that extends Dataset class . It read config , load data from drive and normalise the data , split it into train and val and convert the mask 
    into binary 

    """
    def __init__(self, data_dir, modality='s1', mask_source='s1', split= 'train'):
        self.modality = modality
        self.data_dir = Path(data_dir)
        self.mask_source = mask_source
        self.split = split

        # parse modality string
        if self.modality =='s1':
            self.load_s1 = True
            self.load_s2 = False
        elif self.modality =='s2':
            self.load_s1 = False
            self.load_s2 = True
        elif self.modality =='s1_s2':
            self.load_s1 = True
            self.load_s2 = True
        else:
            raise ValueError(f"Unknown Modality :{modality}. Use s1, s2 or s1_s2 '")
        
        # paths
        self.s1_dir = self.data_dir /"Dataset" /"Sentinel1" /"S1"
        self.s2_dir = self.data_dir /"Dataset" /"Sentinel2" /"S2"
        self.s1_mask_dir = self.data_dir /"Dataset" /"Sentinel1" /"Floodmaps"
        self.s2_mask_dir = self.data_dir /"Dataset" /"Sentinel2" /"Floodmaps"

        # get the file and sort them
        s1_files = sorted([f for f in os.listdir(self.s1_dir) if f.endswith('.tif')])
        s2_files = sorted([f for f in os.listdir(self.s2_dir) if f.endswith('.tif')])

        # For s1_s2, need both; for s1 only need s1; for s2 only need s2
        if modality == "s1":
            all_files = s1_files
        elif modality == "s2":
            all_files = s2_files
        else:  # s1_s2
            common = set(s1_files) & set(s2_files)
            all_files = sorted(list(common))

        # train / Val split (80/20)
        split_idx = int(0.8*len(all_files))
        if split =='train':
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]
        print(f"\n[{split.upper()}] Modality: {modality}")
        print(f"  Files: {len(self.files)}")
        print(f"  Mask source: {mask_source}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,idx):
        filename = self.files[idx]
        image_parts =[]

        if self.load_s1:
            s1 = self._load_s1(filename)
            image_parts.append(s1)
        
        if self.load_s2:
            s2 = self._load_s2(filename)
            image_parts.append(s2)
        
        # concat the images
        images = np.concatenate(image_parts, axis= 0)

        # load mmasks 

        if self.mask_source == 's1':
            mask = self._load_s1_mask(filename)
        else :
            mask = self._load_s2_mask(filename)
        

        return {
            "images": torch.from_numpy(images).float(),
            "masks" : torch.from_numpy(mask).float()

        }

    def _load_s1(self, filename):
        with rasterio.open(self.s1_dir/filename) as src:
            sar = src.read().astype(np.float32)
            #normalise
            sar = np.log1p(np.abs(sar))
            sar = (sar-sar.mean(axis=(1,2),keepdims=True))/(sar.std(axis=(1,2),keepdims=True)+1e-8)
        return sar

    def _load_s2(self, filename):
        with rasterio.open(self.s2_dir/filename) as src:
            optical = src.read().astype(np.float32)
            #normalise
            optical = optical/10000.0
            optical= np.clip(optical, 0,1)
        return optical
    
    def _load_s1_mask(self, filename):
        with rasterio.open(self.s1_mask_dir/filename) as src:
            mask = src.read(1).astype(np.float32)
        return (mask ==1).astype(np.float32)[np.newaxis,:,:]

    def _load_s2_mask(self, filename):
        with rasterio.open(self.s2_mask_dir/filename) as src:
            mask = src.read(1).astype(np.float32)
        return (mask ==1).astype(np.float32)[np.newaxis,:,:]

if __name__ =="__main__":
    config = CONFIG
    dataset= FloodDataset(config['data']['data_dir'], modality=config['data']['modality'],mask_source=config['data']['mask_source'],split='train')
    loader = DataLoader(dataset,batch_size =4,shuffle=True, num_workers=2)
    batch = next(iter(loader))
    print(f"✓ Image shape: {batch['images'].shape}")
    print(f"✓ Mask shape: {batch['masks'].shape}")