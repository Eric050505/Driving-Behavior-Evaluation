import numpy as np
from tqdm import tqdm


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    # converted_len = int(clip_len * frame_sample_rate)
    # end_idx = np.random.randint(clip_len, seg_len)
    # start_idx = end_idx - converted_len
    # indices = np.linspace(start_idx, end_idx, num=clip_len)
    # indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    # return indices
    if seg_len < clip_len * frame_sample_rate:
        indices = np.linspace(0, seg_len - 1, num=clip_len, dtype=int)
    else:
        average_duration = seg_len // clip_len
        offsets = np.multiply(range(clip_len), average_duration) + np.random.randint(average_duration, size=clip_len)
        indices = np.sort(offsets)
    indices = np.clip(indices, 0, seg_len - 1)
    return indices

def download_model():
    from transformers import VivitImageProcessor, VivitForVideoClassification
    from huggingface_hub import snapshot_download
    import os

    # Define local directory to save the model
    local_model_dir = "./vivit-b-16x2-kinetics400"

    # Create directory if it doesn't exist
    os.makedirs(local_model_dir, exist_ok=True)

    # Download the model weights and configuration from Hugging Face Hub
    snapshot_download(repo_id="google/vivit-b-16x2-kinetics400", local_dir=local_model_dir, tqdm_class=tqdm)

    # Load the image processor and model from the local directory
    image_processor = VivitImageProcessor.from_pretrained(local_model_dir)
    model = VivitForVideoClassification.from_pretrained(local_model_dir)

    print(f"Model and processor successfully downloaded and loaded from {local_model_dir}")