import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"

from transformers import Trainer, TrainingArguments, AdamW, VivitConfig
from model_configuration import initialise_model, compute_metrics
from preprocessing import create_dataset
from data_handling import frames_convert_and_create_dataset_dictionary
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

print("Importing libs: Done")
TRAIN_MODE = True
root_dir = "./DBE/ViViT/ViViT-Driving-Scene"


def load_data():
    path_file = f"{root_dir}/data/UCF101_subset"

    video_dict, class_labels = frames_convert_and_create_dataset_dictionary(path_file)
    class_labels = sorted(class_labels)
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    print(f"Unique classes: {list(label2id.keys())}.")
    shuffled_dataset = create_dataset(video_dict, save_path="train_dataset_cache", use_cache=True)
    print(f"Shuffled dataset shape: {shuffled_dataset.shape}")
    print(f"Shuffled dataset labels: {shuffled_dataset['train'].features}")
    return shuffled_dataset, label2id, id2label


def train(shuffled_dataset, device):
    model = initialise_model(shuffled_dataset, device)
    print("Pretrained Model loaded.")
    training_output_dir = "/tmp/results"
    training_args = TrainingArguments(
        output_dir=training_output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-05,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        seed=42,
        evaluation_strategy="steps",
        eval_steps=10,
        warmup_steps=int(0.1 * 20),
        optim="adamw_torch",
        lr_scheduler_type="linear",
        report_to="none"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-05, betas=(0.9, 0.999), eps=1e-08)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_dataset["train"],
        eval_dataset=shuffled_dataset["UCF101_test"],
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics
    )
    print("Start training...")
    train_results = trainer.train()
    trainer.save_model("model")
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

    if TRAIN_MODE:
        shuffled_dataset, label2id, id2label = load_data()
        train(shuffled_dataset, device)

    inference(path_files_val="data/UCF101_test/")


if __name__ == '__main__':
    main()
