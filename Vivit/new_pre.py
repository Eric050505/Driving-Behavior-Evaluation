import os
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, VivitForVideoClassification, VivitImageProcessor, VivitConfig
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

class NpyVideoDataset(TorchDataset):
    def __init__(self, mode="train", root_dir="."):
        self.root_dir = os.path.join(root_dir, mode)
        self.samples = []
        self.labels = []
        self.class_labels = []
        
        for npy_file in tqdm(sorted(os.listdir(self.root_dir)), desc=f"Scanning {mode} files"):
            if not npy_file.endswith('.npy'):
                continue
                
            file_path = os.path.join(self.root_dir, npy_file)
            class_name = os.path.splitext(npy_file)[0]
            
            if class_name not in self.class_labels:
                self.class_labels.append(class_name)
                
            video_data_batch = np.load(file_path, allow_pickle=True)
            for i in range(len(video_data_batch)):
                self.samples.append((file_path, i))
                label = self.class_labels.index(class_name)
                self.labels.append(label)
                
        print(f"{mode} dataset initialized. Samples: {len(self.samples)}, Classes: {len(self.class_labels)}")
        print(f"{mode} labels range: {min(self.labels)} to {max(self.labels)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, sample_idx = self.samples[idx]
        label = self.labels[idx]
        video_data_batch = np.load(file_path, allow_pickle=True)
        video_frames = video_data_batch[sample_idx]
        
        if isinstance(video_frames, dict):
            video_frames = video_frames['video']
        
        if len(video_frames.shape) == 3:
            video_frames = np.expand_dims(video_frames, axis=-1)
            
        inputs = image_processor(list(video_frames), return_tensors='pt')
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'labels': label
        }

def load_npy_data(mode="train", root_dir="."):
    dataset = NpyVideoDataset(mode, root_dir)
    return dataset, dataset.class_labels