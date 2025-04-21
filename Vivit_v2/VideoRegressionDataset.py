# import os
#
# import torch
# import av
# import pandas as pd
# from torch.utils.data import Dataset
# from utils import *
#
#
# class VideoRegressionDataset(Dataset):
#     def __init__(self, video_dir, csv_file, image_processor, clip_len=32, frame_sample_rate=2):
#         self.video_dir = video_dir
#         self.image_processor = image_processor
#         self.clip_len = clip_len
#         self.frame_sample_rate = frame_sample_rate
#
#         # Read CSV file
#         self.df = pd.read_csv(csv_file)
#         # Ensure video files exist in the directory
#         self.video_files = [f for f in os.listdir(video_dir) if f in self.df['Scene'].values]
#         self.df = self.df[self.df['Scene'].isin(self.video_files)]
#
#     def __len__(self):
#         return len(self.video_files)
#
#     def __getitem__(self, idx):
#         # Get video file and label
#         video_file = self.video_files[idx]
#         label = self.df[self.df['Scene'] == video_file]['Value'].values[0]
#         label = label * 9 + 1
#
#         # Load video
#         video_path = os.path.join(self.video_dir, video_file)
#         container = av.open(video_path)
#
#         # Sample frames
#         seg_len = container.streams.video[0].frames
#         indices = sample_frame_indices(clip_len=32, frame_sample_rate=self.frame_sample_rate, seg_len=seg_len)
#         video = read_video_pyav(container, indices)
#
#         # Process video frames
#         inputs = self.image_processor(list(video), return_tensors="pt")
#         pixel_values = inputs['pixel_values'].squeeze(0)  # Shape: (num_frames, C, H, W)
#
#         return {
#             'pixel_values': pixel_values,
#             'label': torch.tensor(label, dtype=torch.float32)
#         }
import os

import torch
import av
import pandas as pd
from torch.utils.data import Dataset
from utils import *
from sklearn.preprocessing import StandardScaler

class VideoRegressionDataset(Dataset):
    def __init__(self, video_dir, csv_file, image_processor, clip_len=32, frame_sample_rate=2, scaler=None, is_train=False):
        self.video_dir = video_dir
        self.image_processor = image_processor
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate

        # Read CSV file
        self.df = pd.read_csv(csv_file)
        # Ensure video files exist in the directory
        self.video_files = [f for f in os.listdir(video_dir) if f in self.df['Scene'].values]
        self.df = self.df[self.df['Scene'].isin(self.video_files)]
        self.raw_labels = self.df[self.df['Scene'].isin(self.video_files)]['Value'].values.astype(float)
        if scaler is not None:
            self.labels = scaler.transform(self.raw_labels.reshape(-1, 1)).flatten()
        else:
            if is_train:
                self.scaler = StandardScaler().fit(self.raw_labels.reshape(-1, 1))
                self.labels = self.scaler.transform(self.raw_labels.reshape(-1, 1)).flatten()
            else:
                raise ValueError("Scaler must be provided for non-training datasets")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        # Get video file and label
        video_file = self.video_files[idx]
        label = self.labels[idx]

        # Load video
        video_path = os.path.join(self.video_dir, video_file)
        container = av.open(video_path)

        # Sample frames
        seg_len = container.streams.video[0].frames
        indices = sample_frame_indices(clip_len=32, frame_sample_rate=self.frame_sample_rate, seg_len=seg_len)
        video = read_video_pyav(container, indices)

        # Process video frames
        inputs = self.image_processor(list(video), return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # Shape: (num_frames, C, H, W)

        return {
            'pixel_values': pixel_values,
            'label': torch.tensor(label, dtype=torch.float32)
        }
