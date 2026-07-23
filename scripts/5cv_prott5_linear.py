"""
ProtT5 + Linear ablation (5-fold CV).

Same data, KFold split (seed=42) and training schedule as ``5cvCaskNet.py``,
but the architecture is reduced to:

  ProtT5 embeddings (B, L, 1024)
    -> mean-pool over the un-padded sequence positions   (B, 1024)
    -> Linear(1024, 2)                                   (B, 2 = logits)

This script does NOT modify any existing file -- it only imports utilities
(``xlsxProteinDataset``, ``MySelf_collate_fullEmbed``, and the metric helpers
from ``metrics_extra``).

Outputs (a fresh new folder per request):
  metrics_5cv_ablation/{fold}_prott5_linear_overall.txt
  metrics_5cv_ablation/prott5_linear_overall_summary.txt
  metrics_5cv_ablation_bylength/{fold}_prott5_linear_bylength.txt
  metrics_5cv_ablation_bylength/prott5_linear_bylength_summary.txt
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from DataSet.xlsxPLMDataSet import xlsxProteinDataset, MySelf_collate_fullEmbed
from metrics_extra import (
    compute_overall_metrics,
    append_summary,
    compute_length_bin_metrics,
    append_length_bin_summary,
)


class Config:
    batch_size = 64
    max_length = 110
    epochs = 20
    patience = 3
    lr = 1e-4
    weight_decay = 1e-5
    weighted = True
    n_folds = 5
    random_state = 42
    embed_dim = 1024
    num_classes = 2


if torch.backends.mps.is_available():
    Config.device = torch.device("mps")
elif torch.cuda.is_available():
    Config.device = torch.device("cuda")
else:
    Config.device = torch.device("cpu")


# --------------------------------- Model ---------------------------------- #


class ProtT5Linear(nn.Module):
    """Mean-pool ProtT5 embeddings (over real positions) -> single Linear."""

    def __init__(self, embed_dim=Config.embed_dim, num_classes=Config.num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (B, L, E) ProtT5 embeddings, padded positions are all zeros.
        mask = (x.abs().sum(dim=-1) != 0).float().unsqueeze(-1)  # (B, L, 1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (x * mask).sum(dim=1) / denom                    # (B, E)
        logits = self.fc(pooled)
        # Return a dummy attention tensor so the (logits, attn) interface used
        # by metrics_extra (model_kind="casknet") is preserved.
        return logits, torch.zeros(x.size(0), 1, device=x.device)


# ----------------------------- Train / loaders ---------------------------- #


def make_loaders(train_subset, val_subset):
    if Config.weighted:
        lengths, labels = [], []
        for idx in range(len(train_subset)):
            emb, lbl, _, _ = train_subset[idx]
            length = (emb.abs().sum(dim=-1) != 0).sum().item()
            lengths.append(length)
            labels.append(int(lbl))
        weights = [
            2.25 if l < 25 and y == 0 else 1.0
            for l, y in zip(lengths, labels)
        ]
        sampler = WeightedRandomSampler(
            torch.DoubleTensor(weights),
            num_samples=len(weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=Config.batch_size,
            sampler=sampler,
            collate_fn=MySelf_collate_fullEmbed,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=Config.batch_size,
            shuffle=True,
            collate_fn=MySelf_collate_fullEmbed,
        )
    val_loader = DataLoader(
        val_subset,
        batch_size=Config.batch_size,
        collate_fn=MySelf_collate_fullEmbed,
    )
    return train_loader, val_loader


def train_one_fold(model, train_loader, val_loader):
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    no_improve = 0
    for epoch in range(Config.epochs):
        model.train()
        with tqdm(train_loader) as pbar:
            for inputs, labels, _ in pbar:
                inputs = inputs.to(Config.device)
                labels = labels.to(Config.device)
                optimizer.zero_grad()
                logits, _ = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs = inputs.to(Config.device)
                labels = labels.to(Config.device)
                logits, _ = model(inputs)
                val_loss += criterion(logits, labels).item()
                n_batches += 1
        val_loss /= max(n_batches, 1)
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
        scheduler.step()
        print(f"  Epoch {epoch+1:02d}, Val Loss: {val_loss:.4f}")
        if no_improve >= Config.patience:
            print(f"  Early stop after epoch {epoch+1}")
            break


# ----------------------------------- Main --------------------------------- #


def main():
    csv_file = "./final_deduplicated_dataset.csv"
    fasta_path = "secretepro.fasta"
    t5_file = "./Embedding/secreteprofull.h5"
    base_dataset = xlsxProteinDataset(
        csv_file, fasta_path, t5_file, max_length=Config.max_length
    )
    kfold = KFold(
        n_splits=Config.n_folds,
        shuffle=True,
        random_state=Config.random_state,
    )

    tag = "prott5_linear"
    overall_summary = f"metrics_5cv_ablation/{tag}_overall_summary.txt"
    bylength_summary = f"metrics_5cv_ablation_bylength/{tag}_bylength_summary.txt"
    for path in (
        overall_summary,
        bylength_summary,
        bylength_summary + ".folds.json",
    ):
        if os.path.exists(path):
            os.remove(path)
    os.makedirs("metrics_5cv_ablation", exist_ok=True)
    os.makedirs("metrics_5cv_ablation_bylength", exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(range(len(base_dataset))), 1
    ):
        print(f"\n[{tag}] Training fold {fold}")
        train_subset = Subset(base_dataset, train_idx)
        val_subset = Subset(base_dataset, val_idx)
        train_loader, val_loader = make_loaders(train_subset, val_subset)
        model = ProtT5Linear().to(Config.device)
        train_one_fold(model, train_loader, val_loader)

        overall_file = f"metrics_5cv_ablation/{fold}_{tag}_overall.txt"
        bylength_file = (
            f"metrics_5cv_ablation_bylength/{fold}_{tag}_bylength.txt"
        )
        overall = compute_overall_metrics(
            val_loader, model, Config, overall_file, model_kind="casknet"
        )
        append_summary(overall_summary, fold, overall)
        bylength = compute_length_bin_metrics(
            val_loader, model, Config, bylength_file, model_kind="casknet"
        )
        append_length_bin_summary(bylength_summary, fold, bylength)


if __name__ == "__main__":
    main()
