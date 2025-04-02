import numpy as np
import av
import torch
from transformers import VivitImageProcessor, VivitModel
from datasets import Dataset, load_from_disk
import os

image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400", local_files_only=True)

def process_example(example):
    inputs = image_processor(list(np.array(example['video'])), return_tensors='pt')
    inputs['labels'] = example['labels']
    return inputs


def load_dataset(save_path="dataset_cache"):
    if os.path.exists(save_path):
        return load_from_disk(save_path)
    else:
        raise FileNotFoundError(f"No saved dataset found at {save_path}")


def create_dataset_generator(temp_dir, class_labels):
    print("Start creating...")
    if not class_labels:
        print("Error when generating dataset, check class_labels")
        return
    for cls in tqdm(class_labels, desc=f"Creating {mode} classes"):
        print(cls)
        class_file = os.path.join(temp_dir, f"{cls}.npy")
        if os.path.exists(class_file):
            try:
                class_data = np.load(class_file, allow_pickle=True)
                for item in class_data:
                    yield item
            except Exception as e:
                print(f"Error loading {class_file}: {e}")


def create_dataset(temp_dir, save_path, class_labels=None, use_cache=True):
    if use_cache and os.path.exists(save_path):
        print("Loading dataset from cache...")
        return load_from_disk(save_path)
    
    dataset = Dataset.from_generator(lambda: create_dataset_generator(temp_dir, class_labels))
    dataset = dataset.class_encode_column("labels")
    processed_dataset = dataset.map(process_example, batched=False)
    processed_dataset = processed_dataset.remove_columns(['video'])
    shuffled_dataset = processed_dataset.shuffle(seed=42)
    shuffled_dataset = shuffled_dataset.map(lambda x: {'pixel_values': torch.tensor(x['pixel_values']).squeeze()})
    shuffled_dataset.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}")
    return shuffled_dataset
