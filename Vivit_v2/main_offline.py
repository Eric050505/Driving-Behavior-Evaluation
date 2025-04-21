import os
import numpy as np
import csv
from torch.utils.data import DataLoader
from transformers import VivitImageProcessor
from VideoRegressionDataset import VideoRegressionDataset
from train import *


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            # outputs = model(pixel_values=pixel_values, labels=labels)
            # loss = outputs.loss.mean()
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            # preds = outputs.logits
            # if preds.dim() == 2:  # [batch_size, 1]
            #     preds = preds.squeeze(-1)
            predictions.extend(logits.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    return avg_loss, predictions, true_labels


def load_data(config, data_dir="data/ori_data", csv_file="data/comfort-class2.csv"):
    image_processor = VivitImageProcessor.from_pretrained("./vivit-b-16x2-kinetics400", local_files_only=True)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    train_dataset = VideoRegressionDataset(train_dir, csv_file, image_processor, is_train=True)
    val_dataset = VideoRegressionDataset(val_dir, csv_file, image_processor, scaler=train_dataset.scaler)
    test_dataset = VideoRegressionDataset(test_dir, csv_file, image_processor, scaler=train_dataset.scaler)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def main():
    checkpoint_dir = "checkpoints_unfreeze_60epoch_titan"
    log_dir = "logs_unfreeze_60epoch_titan"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_header = ["epoch", "train_loss", "val_loss", "val_mae", "test_loss", "test_mae"]

    with open(f"{log_dir}/training_log.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(log_header)

    config = {
        "batch_size": 16,
        "learning_rate": 1e-4,
        "epochs": 180,
        "clip_len": 32,
        "frame_sample_rate": 4
    }

    train_loader, val_loader, test_loader = load_data(config)
    best_val_loss = float('inf')

    model, optimizer, scaler, device = initial_model(config)
    criterion = torch.nn.MSELoss()
    for epoch in range(config["epochs"]):
        unfreeze_layers(model, current_epoch=epoch, total_freeze_epochs=5)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, criterion)

        val_loss, val_preds, val_labels = evaluate(model, val_loader, device, criterion)
        val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_labels)))

        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae
        }
        torch.save(checkpoint, f"{checkpoint_dir}/epoch_{epoch + 1}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")

        with open(f"{log_dir}/training_log.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1,
                             f"{train_loss:.4f}",
                             f"{val_loss:.4f}",
                             f"{val_mae:.4f}",
                             "", ""])

        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

    model.load_state_dict(torch.load(f"{checkpoint_dir}/best_model.pth", weight_only=True))
    test_loss, test_preds, test_labels = evaluate(model, test_loader, device, criterion)
    test_mae = np.mean(np.abs(np.array(test_preds) - np.array(test_labels)))

    with open(f"{log_dir}/training_log.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(["final_test",
                         "", "", "",
                         f"{test_loss:.4f}",
                         f"{test_mae:.4f}"])

    torch.save(model.state_dict(), f"{checkpoint_dir}/final_model.pth")

    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f}")


if __name__ == "__main__":
    main()
