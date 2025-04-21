import os

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import VivitForVideoClassification, VivitImageProcessor

from VideoRegressionDataset import VideoRegressionDataset
from main_offline import evaluate
from train import unfreeze_layers


def test(name):
    criterion = torch.nn.MSELoss()
    data_dir = "data/ori_data"
    csv_file = "data/comfort-class2.csv"
    test_dir = os.path.join(data_dir, "test")
    train_dir = os.path.join(data_dir, "train")
    config = {
        "batch_size": 2,
        "learning_rate": 1e-4,
        "epochs": 30,
        "clip_len": 32,
        "frame_sample_rate": 4
    }
    image_processor = VivitImageProcessor.from_pretrained("./vivit-b-16x2-kinetics400")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VivitForVideoClassification.from_pretrained("./vivit-b-16x2-kinetics400", local_files_only=True)
    unfreeze_layers(model, current_epoch=1, total_freeze_epochs=5)

    model.num_labels = 1
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(model.config.hidden_size, 1)
    )
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(f"{name}/best_model.pth", weights_only=True))

    train_dataset = VideoRegressionDataset(train_dir, csv_file, image_processor, is_train=True)
    test_dataset = VideoRegressionDataset(test_dir, csv_file, image_processor, scaler=train_dataset.scaler)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    test_loss, test_preds, test_labels = evaluate(model, test_loader, device, criterion)
    test_mae = np.mean(np.abs(np.array(test_preds) - np.array(test_labels)))
    result = [[], []]
    for pred, label in zip(test_preds, test_labels):
        print(pred, label)
        result[0].append(pred.item())
        result[1].append(label.item())
    df = pd.DataFrame.from_records(result)
    df.to_csv(f"{name}/results.csv", index=False)
    print(f"Test MAE: {test_mae}")


if __name__ == "__main__":
    test("unfreeze")
