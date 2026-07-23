import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from metrics_extra import (
    compute_overall_metrics,
    append_summary,
    compute_length_bin_metrics,
    append_length_bin_summary,
)

# CNN Config
class Config:
    batch_size = 8
    pad_size = 110
    embed_size = 256
    filter_sizes = (2, 3, 4)
    num_filters = 128
    num_classes = 2
    dropout = 0.5
    lr = 1e-3
    epochs = 15


if torch.cuda.is_available():
    Config.device = torch.device("cuda")
else:
    Config.device = torch.device("cpu")

# CNN Model
class CNN(nn.Module):
    def __init__(self, config, vocab_size):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config.embed_size)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add channel dimension
        x = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Dataset
class ProteinDataset(Dataset):
    def __init__(self, df, vocab, max_length):
        self.df = df
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.iloc[idx]['Sequence']
        label = self.df.iloc[idx]['label']
        seq_idx = [self.vocab.get(c, 0) for c in seq[:self.max_length]]
        return torch.tensor(seq_idx), torch.tensor(label, dtype=torch.long)

# Collate Function
def collate_fn(batch):
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    padded_seqs = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded_seqs, labels

# Training Loop
def train_model(model, train_loader, val_loader, val_dataset, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    for epoch in range(config.epochs):
        model.train()
        with tqdm(train_loader) as pbar:
            for i, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(Config.device)
                labels = labels.to(Config.device)
                optimizer.zero_grad()
                output= model(inputs)
                loss = criterion(output, labels.to(Config.device))
                loss.backward()
                optimizer.step()


    # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for (inputs, labels) in val_loader:
                inputs = inputs.to(Config.device)
                labels = labels.to(Config.device)
                output= model(inputs)
                val_loss += criterion(output, labels.to(Config.device)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(Config.device)).sum().item()
                )
        val_loss /= len(val_loader)
        val_accuracy /= len(val_dataset)

    # Update learning rate
        scheduler.step()

        print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
        )



def calculate_interval_metrics(val_loader, model, config, output_file="metrics.txt"):
    model.eval()
    interval_stats = defaultdict(lambda: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0})
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            seq_lengths = (inputs != 0).sum(dim=1).cpu().numpy()
            for i in range(len(labels)):
                true, pred = labels[i].item(), preds[i].item()
                length = seq_lengths[i]
                start = max(10, (length // 5) * 5)
                end = min(start + 4, config.pad_size)
                interval = (start, end)
                if true == 1:
                    if pred == 1:
                        interval_stats[interval]['TP'] += 1
                    else:
                        interval_stats[interval]['FN'] += 1
                else:
                    if pred == 0:
                        interval_stats[interval]['TN'] += 1
                    else:
                        interval_stats[interval]['FP'] += 1

    results = []
    with open(output_file, 'w') as f:
        f.write("Interval\tSensitivity\tSpecificity\n")
        for interval in sorted(interval_stats):
            tp, fn = interval_stats[interval]['TP'], interval_stats[interval]['FN']
            tn, fp = interval_stats[interval]['TN'], interval_stats[interval]['FP']
            sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f.write(f"[{interval[0]}-{interval[1]}]\t{sen:.4f}\t{spec:.4f}\n")
            results.append((interval, sen, spec))
    return results

# Main Script
if __name__ == "__main__":
    df = pd.read_csv("final_deduplicated_dataset.csv")
    df = df.sample(frac=1, random_state=42)
    all_chars = set("".join(df['Sequence']))
    vocab = {c: i+1 for i, c in enumerate(all_chars)}
    vocab_size = len(vocab) + 1

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
        print(f"Training fold {fold}")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        print(train_df)
        print(val_df)

        train_dataset = ProteinDataset(train_df, vocab, Config.pad_size)
        val_dataset = ProteinDataset(val_df, vocab, Config.pad_size)

        train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, collate_fn=collate_fn)

        config = Config
        model = CNN(config, vocab_size).to(config.device)
        train_model(model, train_loader, val_loader, val_dataset, config)
        calculate_interval_metrics(val_loader, model, config, output_file=f"5cv/{fold}_cnn.txt")
        overall = compute_overall_metrics(
            val_loader, model, config, f"metrics_5cv/{fold}_cnn_overall.txt", model_kind="cnn"
        )
        append_summary("metrics_5cv/cnn_overall_summary.txt", fold, overall)

        bylength = compute_length_bin_metrics(
            val_loader, model, config,
            f"metrics_5cv_bylength/{fold}_cnn_bylength.txt",
            model_kind="cnn",
        )
        append_length_bin_summary(
            "metrics_5cv_bylength/cnn_bylength_summary.txt", fold, bylength
        )
