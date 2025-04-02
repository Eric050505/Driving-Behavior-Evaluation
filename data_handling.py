import os
import numpy as np
import av
from tqdm import tqdm
from pathlib import Path


def generate_all_files(root: Path, only_files: bool = True):
    for p in root.rglob("*"):
        if only_files and not p.is_file():
            continue
        yield p
        
def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            reformatted_frame = frame.reformat(width=224,height=224)
            frames.append(reformatted_frame)
    new=np.stack([x.to_ndarray(format="rgb24") for x in frames])

    return new


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def frames_convert_and_create_dataset_dictionary(directory, dataset="UCF101", mode="train"):
    class_labels = []
    all_videos=[]
    video_files= []
    sizes = []
    for p in generate_all_files(Path(directory), only_files=True):

        if dataset == "UCF101" and p.name.endswith(".avi"):
            set_files = str(p).split("/")[2] # train, val or test
            cls = str(p).split("/")[3] # class
            file= str(p).split("/")[4] # file name
            file_name= os.path.join(directory, set_files, cls, file)
        elif dataset == "k400" and p.name.endswith(".mp4"):
            set_files = str(p).split("/")[2] # train, val or test
            if mode == "train" and set_files == "test":
                continue
            if mode == "test" and (set_files == "val" or set_files == "train"):
                continue
            cls = str(p).split("/")[3] # class
            file= str(p).split("/")[4] # file name
            file_name= os.path.join(directory, set_files, cls, file)
        else:
            continue
        
        # print(f"Processing file {file_name}")    
        # Process class
        if cls not in class_labels:
            class_labels.append(cls)
            print(cls)
        # process video File
        try:
            container = av.open(file_name)
            # print(f"Processing file {file_name} number of Frames: {container.streams.video[0].frames}")  
            indices = sample_frame_indices(clip_len=10, frame_sample_rate=1,seg_len=container.streams.video[0].frames)
            video = read_video_pyav(container=container, indices=indices)
            all_videos.append({'video': video, 'labels': cls})
            sizes.append(container.streams.video[0].frames)
        except (av.error.InvalidDataError, IndexError, ValueError) as e:
            print(f"SKIPPING AND DELETING {file_name} due to error: {e}")
            try:
                os.remove(file_name)
                print(f"Deleted {file_name}")
            except OSError as delete_error:
                print(f"Failed to delete {file_name}: {delete_error}")
        
    sizes = np.array(sizes)
    print(f"Min number frames {sizes.min()}")
    return all_videos, class_labels

def frames_convert(directory, mode="train", output_dir="temp_video_data"):
    class_labels = []
    sizes = []
    os.makedirs(output_dir, exist_ok=True)

    mode_dir = os.path.join(directory, mode)
    if not os.path.exists(mode_dir):
        print(f"Error: {mode_dir} not exist")
        return class_labels

    class_folders = [f for f in os.listdir(mode_dir) if os.path.isdir(os.path.join(mode_dir, f))]
    class_folders.sort()
    
    # mid_point = len(class_folders) // 2
    # class_folders = class_folders[:mid_point]
    # class_folders = class_folders[300:350]
    
    
    for cls in tqdm(class_folders, desc=f"Processing {mode} classes"):
        class_dir = os.path.join(mode_dir, cls)
        class_file = os.path.join(output_dir, f"{cls}.npy")
        if os.path.exists(class_file):
            # print(f"Skip: {cls} (Exist: {class_file})")
            if cls not in class_labels:
                class_labels.append(cls)
            continue
        
        if cls not in class_labels:
            class_labels.append(cls)

        video_files = [f for f in os.listdir(class_dir) if f.endswith(".mp4")]
        class_data = []

        for file in video_files:
            file_name = os.path.join(class_dir, file)
            try:
                container = av.open(file_name)
                indices = sample_frame_indices(clip_len=10, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
                video = read_video_pyav(container=container, indices=indices)
                class_data.append({'video': video, 'labels': cls})
                sizes.append(container.streams.video[0].frames)
            except (av.error.InvalidDataError, IndexError, ValueError) as e:
                print(f"SKIPPING AND DELETING {file_name}.")

        if class_data:
            class_file = os.path.join(output_dir, f"{cls}.npy")
            np.save(class_file, class_data)
            print(f"Saved {len(class_data)} videos for class {cls} to {class_file}")

    sizes = np.array(sizes)
    if sizes.size > 0:
        print(f"Min number frames {sizes.min()}")
    return class_labels