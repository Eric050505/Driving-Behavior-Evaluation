import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import VivitImageProcessor, VivitForVideoClassification
from VideoRegressionDataset import VideoRegressionDataset
import wandb


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log batch-level metrics to W&B
        wandb.log({"batch_loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            preds = outputs.logits.squeeze().cpu().numpy()
            predictions.extend(preds.reshape(-1) if preds.ndim == 0 else preds)
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    return avg_loss, predictions, true_labels


def main():
    # Initialize W&B
    wandb.login(key="3f94e5e3f7839f537ae739bcf420e46de992ce6b")
    wandb.init(
        project="dbe-vivit-regression",
        name="dbe-vivit-regression",
        config={
            "batch_size": 2,
            "learning_rate": 1e-4,
            "epochs": 15,
            "clip_len": 32,
            "frame_sample_rate": 4
        })

    # Paths
    data_dir = "data/ori_data"
    csv_file = "data/comfort-class2.csv"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Load image processor and model
    image_processor = VivitImageProcessor.from_pretrained("./vivit-b-16x2-kinetics400",
                                                          local_files_only=True
                                                          )
    model = VivitForVideoClassification.from_pretrained("./vivit-b-16x2-kinetics400",
                                                        local_files_only=True
                                                        )

    # Modify model for regression
    model.num_labels = 1
    model.classifier = torch.nn.Linear(model.config.hidden_size, 1)

    # Create datasets
    train_dataset = VideoRegressionDataset(train_dir, csv_file, image_processor, is_train=True)
    val_dataset = VideoRegressionDataset(val_dir, csv_file, image_processor, scaler=train_dataset.scaler)
    test_dataset = VideoRegressionDataset(test_dir, csv_file, image_processor, scaler=train_dataset.scaler)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False)

    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    # Training loop
    for epoch in range(wandb.config.epochs):
        train_loss = train_one_epoch(model, test_loader, optimizer, device)
        val_loss, val_predictions, val_labels = evaluate(model, val_loader, device)

        # Log epoch-level metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": np.mean(np.abs(np.array(val_predictions) - np.array(val_labels)))
        })

        print(f"Epoch {epoch + 1}/{wandb.config.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Evaluate on test set
    test_loss, test_predictions, test_labels = evaluate(model, test_loader, device)
    wandb.log({
        "test_loss": test_loss,
        "test_mae": np.mean(np.abs(np.array(test_predictions) - np.array(test_labels)))
    })

    print(
        f"Test Loss: {test_loss:.4f}, Test MAE: {np.mean(np.abs(np.array(test_predictions) - np.array(test_labels))):.4f}")

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
