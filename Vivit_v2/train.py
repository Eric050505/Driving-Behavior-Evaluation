import torch
from transformers import VivitForVideoClassification


def _print_model_status(model):
    printed = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer = str(name).split('.')[4] if len(name.split('.')) > 4 else "classifier"
            printed.add(layer)
            if layer not in printed:
                print(f"Trainable Parameter Layer: {layer}")


def unfreeze_layers(model, current_epoch, total_freeze_epochs=5):
    if current_epoch < total_freeze_epochs:
        for name, param in model.named_parameters():
            if not name.startswith(
                    ("module.vivit.encoder.layer.11.", "module.vivit.encoder.layer.10", "module.classifier")):
                param.requires_grad = False
            else:
                print(f"Unfreeze parameters: {name}")
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
    _print_model_status(model)


def train_one_epoch(model,
                    train_loader,
                    optimizer,
                    device,
                    scaler,
                    criterion=torch.nn.MSELoss()
                    ):
    model.train()
    total_loss = 0
    for batch in train_loader:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(pixel_values=pixel_values)
            # loss = outputs.loss.mean()
            logits = outputs.logits  # shape: [batch_size,1] -> [batch_size]
            if logits.dim() == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(-1)

            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def initial_model(config):
    model = VivitForVideoClassification.from_pretrained("./vivit-b-16x2-kinetics400", local_files_only=True)
    unfreeze_layers(model, current_epoch=1, total_freeze_epochs=5)

    model.num_labels = 1
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(model.config.hidden_size, 1)
    )
    model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = [
        {"params": [], "lr": config["learning_rate"] * 0.1, "weight_decay": 0.01},  # bottom
        {"params": [], "lr": config["learning_rate"], "weight_decay": 0.01}  # top
    ]
    for name, param in model.named_parameters():
        if "encoder.layer.11" in name or "encoder.layer.10" in name or "classifier" in name:
            params[1]["params"].append(param)
        else:
            params[0]["params"].append(param)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=0.01
    )
    scaler = torch.amp.GradScaler(device='cuda')
    return model, optimizer, scaler, device
