import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"

from transformers import Trainer, TrainingArguments, AdamW, VivitConfig
from model_configuration import initialise_model, compute_metrics
from new_pre import load_npy_data
from data_handling import frames_convert
import cv2
import av
from data_handling import sample_frame_indices, read_video_pyav
import moviepy.editor
import torch
from safetensors.torch import load_file
from data_handling import generate_all_files
from pathlib import Path
from transformers import VivitImageProcessor, VivitForVideoClassification
from sklearn.metrics import classification_report
from transformers import EarlyStoppingCallback

print("Importing libs: Done")
root_dir = ".."
# root_dir = "./DBE/ViViT/ViViT-Driving-Scene"

def load_data(mode):
    path_file = f"{root_dir}/k400/ori_data"
    class_labels = frames_convert(path_file, mode=mode, output_dir=f"{root_dir}/k400/bin_data/{mode}")
    print("Convert frames: Done")
    dataset = create_dataset(temp_dir=f"{path_file}/../temp/{mode}", save_path=f"{path_file}/../arrow_data/{mode}", class_labels=class_labels, use_cache=False)
    print(f"Shuffled dataset shape: {dataset.shape}")
    print(f"Shuffled dataset labels: {dataset.features}")
    return dataset


def train(train_dataset, val_dataset, device):
    model = initialise_model(val_dataset, device)
    print("Pretrained Model loaded.")
    print(f"Model num_labels: {model.config.num_labels}")
    print(f"Train dataset classes: {len(train_dataset.class_labels)}")
    print(f"Train dataset samples: {len(train_dataset.samples)}")
    print(f"Train dataset labels range: {min(train_dataset.labels)} to {max(train_dataset.labels)}")
    print(f"Val dataset classes: {len(val_dataset.class_labels)}")
    print(f"Val dataset samples: {len(val_dataset.samples)}")
    print(f"Val dataset labels range: {min(val_dataset.labels)} to {max(val_dataset.labels)}")
    training_output_dir = "/tmp/results"
    training_args = TrainingArguments(
        output_dir=training_output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-04,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        seed=42,
        evaluation_strategy="steps",
        eval_steps=500,
        warmup_steps=500,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_accumulation_steps=4,
        fp16=True,                    
        max_grad_norm=1.0,            
        save_strategy="steps",        
        save_steps=500,               
        load_best_model_at_end=True,  
        metric_for_best_model="accuracy",
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-05, betas=(0.9, 0.999), eps=1e-08)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics
    )
    print("Start training...")
    train_results = trainer.train()
    trainer.save_model("model/k400")
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()


def inference(path_files_val):
    video_dict_val, class_labels_val = frames_convert_and_create_dataset_dictionary(path_files_val)
    val_dataset = create_dataset(video_dict_val, save_path="test_dataset_cache", use_cache=False)

    class_labels = []
    true_labels = []
    predictions = []
    predictions_labels = []
    all_videos = []
    video_files = []
    sizes = []
    i = 0

    model_path = "./model" 

    labels = val_dataset['train'].features['labels'].names
    config = VivitConfig.from_pretrained(model_path)
    config.num_classes = len(labels)
    config.id2label = {str(i): c for i, c in enumerate(labels)}
    config.label2id = {c: str(i) for i, c in enumerate(labels)}
    config.num_frames = 10
    config.video_size = [10, 224, 224]

    image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    fine_tune_model = VivitForVideoClassification.from_pretrained(model_path, config=config)

    directory = "./data/UCF101_test"
    for p in generate_all_files(Path(directory), only_files=True):
        set_files = str(p).split("/")[2]  # train or UCF101_test
        cls = str(p).split("/")[3]  # class
        file = str(p).split("/")[4]  # file name
        # file name path
        file_name = os.path.join(directory, set_files, cls, file)
        true_labels.append(cls)
        # Process class
        if cls not in class_labels:
            class_labels.append(cls)
        # process video File
        container = av.open(file_name)
        # print(f"Processing file {file_name} number of Frames: {container.streams.video[0].frames}")
        indices = sample_frame_indices(clip_len=10, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)
        inputs = image_processor(list(video), return_tensors="pt")
        with torch.no_grad():
            outputs = fine_tune_model(**inputs)
            logits = outputs.logits

        # model predicts one of the 400 Kinetics-400 classes
        predicted_label = logits.argmax(-1).item()
        prediction = fine_tune_model.config.id2label[str(predicted_label)]
        predictions.append(prediction)
        predictions_labels.append(predicted_label)
        print(f"file {file_name} True Label {cls}, predicted label {prediction}")

    report = classification_report(true_labels, predictions)
    print(report)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    val_dataset, _ = load_npy_data("val", f"{root_dir}/k400/bin_data/")
    train_dataset, _ = load_npy_data("train", f"{root_dir}/k400/bin_data")
    train(train_dataset, val_dataset, device)

if __name__ == '__main__':
    main()
