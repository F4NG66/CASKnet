from model.kan import KAN
# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from DataSet.xlsxPLMDataSet import xlsxProteinDataset,MySelf_collate_fullEmbed
from torch.utils.data  import Dataset, Subset ,random_split
from sklearn.model_selection  import KFold
import torch.backends
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from torch.utils.data import WeightedRandomSampler

class Config:
    batch_size = 32
    max_length = 110  
    embed_dim = 1024  
    lr = 0.0001
    epochs = 35

if torch.backends.mps.is_available():
    Config.device = torch.device("mps")
elif torch.cuda.is_available():
    Config.device = torch.device("cuda")
else:
    Config.device = torch.device("cpu")

def MySelf_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[
    torch.Tensor, torch.Tensor]:
    embeddings = [item[0] for item in batch]
    label = torch.tensor([item[1] for item in batch])
    embeddings = pad_sequence(embeddings, batch_first=True)
    return embeddings, label


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
        # seq = seq.upper().replace('U','X').replace('Z','X').replace('O','X') 
        seq_idx = [self.vocab.get(c, 0) for c in seq[:self.max_length]]
        return torch.tensor(seq_idx), torch.tensor(label, dtype=torch.long)

class ProteinNameDataset(Dataset):
    def __init__(self, df, vocab, max_length):
        self.df = df
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df.iloc[idx]['Entry']
        seq = self.df.iloc[idx]['Sequence']
        label = self.df.iloc[idx]['label']
        seq_idx = [self.vocab.get(c, 0) for c in seq[:self.max_length]]
        return torch.tensor(seq_idx), torch.tensor(label, dtype=torch.long), name
    

def calculate_interval_metrics(val_loader, model, config, output_file="metrics.txt"):
    model.eval()
    interval_stats = defaultdict(lambda: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0})
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels, _ = batch  
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs, _ = model(inputs)
            seq_lengths = (inputs.abs().sum(dim=-1) != 0).sum(dim=1).cpu().numpy()
            predictions = torch.argmax(outputs, dim=1)
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predictions[i].item()
                length = int(seq_lengths[i].item()) 
                start = max(10, (length // 5) * 5)  
                end = min(start + 4, 110)
                interval = (start, end)
                if true_label == 1:
                    if pred_label == 1:
                        interval_stats[interval]['TP'] += 1
                    else:
                        interval_stats[interval]['FN'] += 1
                else:
                    if pred_label == 0:
                        interval_stats[interval]['TN'] += 1
                    else:
                        interval_stats[interval]['FP'] += 1
    results = []
    for interval in sorted(interval_stats.keys()):
        stats = interval_stats[interval]
        tp, fn = stats['TP'], stats['FN']
        tn, fp = stats['TN'], stats['FP']
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results.append((interval, sensitivity, specificity))

    with open(output_file, 'w') as f:
        f.write("Interval\tSensitivity\tSpecificity\n")
        for (start, end), sen, spec in results:
            f.write(f"[{start}-{end}]\t{sen:.4f}\t{spec:.4f}\n")
    print(f"Interval-based metrics saved to {output_file}")
    return results



df = pd.read_csv("final_deduplicated_dataset.csv")

all_chars = set("".join(df['Sequence']))
vocab = {'A': 9, 'B': 5, 'C': 3, 'D': 7, 'E': 24, 'F': 23, 'G': 20, 'H': 17, 'I': 21, 'K': 4, 'L': 12, 'M': 10, 'N': 22, 'P': 6, 'Q': 13, 'R': 19, 'S': 11, 'T': 15, 'U': 16, 'V': 2, 'W': 14, 'X': 18, 'Y': 8, 'Z': 1}
vocab_size = len(vocab)


csv_file = "./final_deduplicated_dataset.csv"
fasta_path ="secretepro.fasta"
t5_file = "./Embedding/secreteprofull.h5"
xlsxProteinDataset = xlsxProteinDataset(csv_file, fasta_path, t5_file, max_length=110)

train_size = int(0.8 * len(xlsxProteinDataset))
test_size = len(xlsxProteinDataset) - train_size
train_dataset, test_dataset = random_split(xlsxProteinDataset, [train_size, test_size])
weighted=True
if weighted:
    lengths = []  
    labels = []  
    for idx in range(len(train_dataset)):
        seq_tensor, label, _ = train_dataset[idx]
        labels.append(label.item())
        if seq_tensor.dim() == 2:
            length = (seq_tensor.abs().sum(dim=-1) != 0).sum().item()
        else:
            length = (seq_tensor != 0).sum().item()
        lengths.append(length)

    weights = [2.0 if l < 25 and lbl == 0 else 1.0 for l, lbl in zip(lengths, labels)]

    weights_tensor = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, sampler=sampler, collate_fn=MySelf_collate_fullEmbed)
else:
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=MySelf_collate_fullEmbed)
val_loader = DataLoader(test_dataset, batch_size=Config.batch_size,collate_fn = MySelf_collate_fullEmbed )
Save_dataloader = DataLoader(xlsxProteinDataset, batch_size=1 )
# Define model
model = KAN([4*64, 64, 2])
model.to(Config.device)
# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# Define loss
criterion = nn.CrossEntropyLoss()

train_losses = []
val_losses = []

for epoch in range(3):
    # Train
    model.train()
    with tqdm(train_loader) as pbar:
        for i, (inputs, labels,_) in enumerate(pbar):
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device)
            optimizer.zero_grad()
            output,_ = model(inputs)
            loss = criterion(output, labels.to(Config.device))
            loss.backward()
            optimizer.step()


    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for (inputs, labels,_) in val_loader:
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device)
            output,_ = model(inputs)
            val_loss += criterion(output, labels.to(Config.device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(Config.device)).sum().item()
            )
    val_loss /= len(val_loader)
    val_accuracy /= len(test_dataset)

    # Update learning rate
    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
    )

interval_output_file = f"5cv/{1}CASKnet.txt"
calculate_interval_metrics(val_loader, model, Config, output_file=interval_output_file)

