import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dataset import TransformerDataset
from transformer_timeseries import TimeSeriesTransformer
from inference import run_encoder_decoder_inference

# 超参数设置
enc_seq_len = 168  # 编码器输入序列长度（过去7天）
dec_seq_len = 48  # 解码器输入序列长度（预测窗口）
target_seq_len = 48  # 目标序列长度（预测48小时）
batch_size = 32
epochs = 10
learning_rate = 0.001
input_size = 1  # 单变量时间序列
num_predicted_features = 1  # 预测一个变量
dim_val = 512  # Transformer 的 d_model
n_heads = 8  # 注意力头数
n_encoder_layers = 4
n_decoder_layers = 4


def prepare_data(file_path):
    df = pd.read_csv(file_path)
    data = df['FCR_N_PriceEUR'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    data_tensor = torch.FloatTensor(data_normalized)
    return data_tensor, scaler


def generate_indices(data_length, enc_seq_len, target_seq_len):
    indices = []
    for i in range(data_length - enc_seq_len - target_seq_len + 1):
        indices.append((i, i + enc_seq_len + target_seq_len))
    return indices


def split_dataset(data_tensor, train_ratio=0.7, val_ratio=0.15):
    total_length = len(data_tensor)
    train_end = int(total_length * train_ratio)
    val_end = int(total_length * (train_ratio + val_ratio))
    train_data = data_tensor[:train_end]
    val_data = data_tensor[train_end:val_end]
    test_data = data_tensor[val_end:]
    return train_data, val_data, test_data


def generate_square_subsequent_mask(dim1, dim2):
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

def main():
    file_path = "../data/dfs_merged_upload.csv"

    data_tensor, scaler = prepare_data(file_path)

    train_data, val_data, test_data = split_dataset(data_tensor)
    print(f"Train length: {len(train_data)}, Val length: {len(val_data)}, Test lengths: {len(test_data)}")

    train_indices = generate_indices(len(train_data), enc_seq_len, target_seq_len)
    val_indices = generate_indices(len(val_data), enc_seq_len, target_seq_len)
    test_indices = generate_indices(len(test_data), enc_seq_len, target_seq_len)

    train_dataset = TransformerDataset(train_data, train_indices, enc_seq_len, dec_seq_len, target_seq_len)
    val_dataset = TransformerDataset(val_data, val_indices, enc_seq_len, dec_seq_len, target_seq_len)
    test_dataset = TransformerDataset(test_data, test_indices, enc_seq_len, dec_seq_len, target_seq_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TimeSeriesTransformer(
        input_size=input_size,
        dec_seq_len=dec_seq_len,
        batch_first=True,
        out_seq_len=target_seq_len,
        dim_val=dim_val,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_heads=n_heads,
        num_predicted_features=num_predicted_features
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: ", device)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for _, (src, tgt, tgt_y) in enumerate(train_dataloader):
            src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

            optimizer.zero_grad()
            tgt_mask = generate_square_subsequent_mask(dec_seq_len, dec_seq_len).to(device)
            src_mask = generate_square_subsequent_mask(dec_seq_len, enc_seq_len).to(device)

            prediction = model(src, tgt, src_mask, tgt_mask)
            prediction = torch.squeeze(prediction, dim=-1)
            loss = criterion(prediction, tgt_y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.6f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (src, _, tgt_y) in enumerate(val_dataloader):
                src, tgt_y = src.to(device), tgt_y.to(device)
                prediction = run_encoder_decoder_inference(
                    model,
                    src,
                    target_seq_len,
                    batch_size=src.shape[0],
                    device=device,
                    batch_first=True
                )
                prediction = torch.squeeze(prediction, dim=-1)
                loss = criterion(prediction, tgt_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.6f}")

    torch.save(model.state_dict(), "transformer_model.pth")
    print("Model saved transformer_model.pth")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (src, _, tgt_y) in enumerate(test_dataloader):
            src, tgt_y = src.to(device), tgt_y.to(device)
            prediction = run_encoder_decoder_inference(
                model, src, target_seq_len, batch_size=src.shape[0], device=device, batch_first=True
            )
            prediction = torch.squeeze(prediction, dim=-1)
            loss = criterion(prediction, tgt_y)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.6f}")


if __name__ == "__main__":
    main()