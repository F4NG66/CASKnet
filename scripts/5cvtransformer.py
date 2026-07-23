import random
from typing import List, Tuple
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from Bio.SeqUtils import seq3
from matplotlib.colors  import Normalize
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from Bio.SeqUtils import seq3
from matplotlib.colors  import Normalize
from collections import defaultdict
from metrics_extra import (
    compute_overall_metrics,
    append_summary,
    compute_length_bin_metrics,
    append_length_bin_summary,
)

# 配置参数
class Config:
    batch_size = 32
    max_length = 110 
    embed_dim = 128  
    num_heads = 4  
    ff_dim = 256  
    num_layers = 1 
    num_classes = 2  
    lr = 0.001
    epochs = 15
    
if torch.backends.mps.is_available():
    Config.device = torch.device("mps")
elif torch.cuda.is_available():
    Config.device = torch.device("cuda")
else:
    Config.device = torch.device("cpu")

def MySelf_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[
    torch.Tensor, torch.Tensor]:
    embeddings = [item[0] for item in batch]
    secretion = torch.tensor([item[1] for item in batch])
    embeddings = pad_sequence(embeddings, batch_first=True)
    return embeddings, secretion

# 自定义数据集类
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
        # 将序列转换为索引
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

class ProteinTransformer(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embed_dim)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, config.max_length, config.embed_dim))
        #learnable position parameters
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.classifier = nn.Linear(config.embed_dim, config.num_classes)
        self.attention_weights = None  # 存储注意力权重

    def forward(self, x):

        self.attention_weights = []
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]  # Setting Position Size
        for layer in self.transformer.layers:
            x = layer(x)
            self.attention_weights.append(layer.attention_weights)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = None

    def _sa_block(self, x, attn_mask =None, key_padding_mask =None,is_causal = False):
        x, weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.attention_weights = weights
        return self.dropout1(x)

def train_model(model, train_loader, val_loader, val_dataset, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    #plt.ion()  # 启用交互模式
    train_losses = []
    val_losses = []
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # 验证
        model.eval()
        val_loss = 0
        correctCount = 0
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            val_pred = torch.argmax(outputs, dim=1)
            correctCount += (val_pred == labels.data).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f}\n")
        print(f"val Accuracy:", correctCount / len(val_dataset), '\n')
        # 实时更新曲线（添加在epoch循环末尾）
        #plt.clf()  # 清除旧图
        #plt.plot(train_losses, 'b-', label='Training Loss')
        #plt.plot(val_losses, 'r--', label='Validation Loss')
        #plt.title(f'Loss  Curve (Epoch {epoch + 1})')
        #plt.xlabel('Epoch')
        #plt.ylabel('Loss')
        #plt.legend()
        #plt.grid(True)
        #plt.pause(0.1)  # 短暂暂停让图像更新
    #plt.ioff()  # 关闭交互模式
    #plt.show() 
    #plt.ioff()  
    #plt.close() 

def calculate_interval_metrics(val_loader, model, config, output_file="metrics.txt"):
    model.eval()
    interval_stats = defaultdict(lambda: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0})
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            seq_lengths = (inputs != 0).sum(dim=1).cpu().numpy()
            predictions = torch.argmax(outputs, dim=1)
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predictions[i].item()
                length = seq_lengths[i]
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
    # 写入文件
    with open(output_file, 'w') as f:
        f.write("Interval\tSensitivity\tSpecificity\n")
        for (start, end), sen, spec in results:
            f.write(f"[{start}-{end}]\t{sen:.4f}\t{spec:.4f}\n")
    print(f"Metrics saved to {output_file}")
    return results



# 主流程
if __name__ == "__main__":
    random_repeat=False
    df = pd.read_csv("final_deduplicated_dataset.csv")
    df = df.sample(frac=1, random_state=42)
    # vocab list
    all_chars = set("".join(df['Sequence']))
    vocab = {c: i + 1 for i, c in enumerate(all_chars)}  # padding with 0
    vocab_size = len(vocab) + 1
    all_metrics = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
        if random_repeat:
            train_df, val_df = train_test_split(df, test_size=0.2)
            print(train_df)
            print(val_df)
        else:
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            print(train_df)
            print(val_df)

        train_dataset = ProteinDataset(train_df, vocab, Config.max_length)
        val_dataset = ProteinDataset(val_df, vocab, Config.max_length)

        train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True,collate_fn = MySelf_collate)
        val_loader = DataLoader(val_dataset, batch_size=Config.batch_size,collate_fn = MySelf_collate )
    
        config = Config()
        model = ProteinTransformer(config, vocab_size).to(config.device)
        train_model(model, train_loader, val_loader,val_dataset, config)

        metrics = calculate_interval_metrics(
         val_loader=val_loader,
         model=model,
         config=config,
         output_file=f"5cv/{fold}_transformer.txt"
        )

        overall = compute_overall_metrics(
            val_loader, model, config,
            f"metrics_5cv/{fold}_transformer_overall.txt",
            model_kind="transformer",
        )
        append_summary("metrics_5cv/transformer_overall_summary.txt", fold, overall)

        bylength = compute_length_bin_metrics(
            val_loader, model, config,
            f"metrics_5cv_bylength/{fold}_transformer_bylength.txt",
            model_kind="transformer",
        )
        append_length_bin_summary(
            "metrics_5cv_bylength/transformer_bylength_summary.txt", fold, bylength
        )





