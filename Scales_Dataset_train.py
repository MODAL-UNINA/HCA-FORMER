#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Lun Lug 07 15:55:04 2025

@author: modal
"""

# %%

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(
    description="Hierarchical Transformer with Multi-Head Self-Attention for Sensor Data"
)
parser.add_argument(
    "--preprocessing_path",
    type=str,
    default="./preprocessed_data",
    help="Path to the dataset CSV file",
)
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument(
    "--n_folds",
    type=int,
    default=10,
    help="Number of folds for K-Fold Cross-Validation",
)
parser.add_argument(
    "--num_heads", type=int, default=8, help="Number of attention heads"
)
parser.add_argument("--depth", type=int, default=6, help="Number of transformer blocks")
parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument(
    "--batch_size", type=int, default=256, help="Batch size for training"
)
parser.add_argument(
    "--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer"
)
parser.add_argument(
    "--weight_decay", type=float, default=1e-5, help="Weight decay for the optimizer"
)
parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
args, _ = parser.parse_known_args()

# ✅ Load data
data_path = Path(args.preprocessing_path).resolve()
df = pd.read_csv(data_path / "Scales_Dataset_processed.csv")

# ✅ Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hier_cols = ["SurvCity", "SurvCountry", "Season", "SeasonDesc", "SensType"]
target_col = "StateTSVF"

# ✅ Encode hierarchical features
label_encoders = {}
for col in hier_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ✅ Drop missing target values and convert to int
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)

# ✅ Extract features
sensor_cols = df.drop(columns=[target_col] + hier_cols).columns.tolist()
X_sensors = df[sensor_cols].values
X_hier = df[hier_cols].values.astype(int)
y = df[target_col].values

# ✅ Combine and resample (no SMOTE here)
X_combined = np.hstack((X_sensors, X_hier))
X_resampled, y_resampled = X_combined, y

# ✅ Split back
X_sensors_resampled = X_resampled[:, : X_sensors.shape[1]]
X_hier_resampled = X_resampled[:, X_sensors.shape[1] :].astype(int)


# ✅ Dataset class
class SensorDataset(Dataset):
    def __init__(self, X_sensors, X_hier, y):
        self.X_sensors = torch.tensor(X_sensors, dtype=torch.float32)
        self.X_hier = torch.tensor(X_hier, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_sensors[idx], self.X_hier[idx], self.y[idx]


# ✅ Transformer components
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_output)
        return self.norm(x)


class Attention_Hierarchical(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.qkv_layers = nn.ModuleList([nn.Linear(dim, dim * 3) for _ in range(5)])
        self.proj_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(5)])
        self.dropout = nn.Dropout(dropout)

    def apply_attention(self, x, qkv_layer, proj_layer):
        qkv = qkv_layer(x)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = torch.sqrt(
            torch.tensor(q.shape[-1], dtype=torch.float32, device=q.device)
        )
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        return proj_layer(self.dropout(x))

    def forward(self, x):
        x = self.mhsa(x)
        parts = [x[:, i, :].unsqueeze(1) for i in range(5)]
        for i in range(5):
            parts[i] = self.apply_attention(
                parts[i], self.qkv_layers[i], self.proj_layers[i]
            )
            if i > 0:
                parts[i] += parts[i - 1]
        return self.dropout(torch.cat(parts, dim=1))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.hta = Attention_Hierarchical(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        original = x
        x = self.mhsa(x)
        if x.shape[1] > 5:
            x = x[:, 1:, :]
        x = self.hta(self.norm(x))
        if original.shape[1] > x.shape[1]:
            x = torch.cat([original[:, 0, :].unsqueeze(1), x], dim=1)
        return x + self.mlp(self.norm(x))


# ✅ Final model
class HierarchicalTransformerWithMHSA(nn.Module):
    def __init__(
        self, num_sensors, num_classes, num_heads=8, depth=6, embed_dim=128, dropout=0.1
    ):
        super().__init__()
        self.tokenizer = nn.Linear(num_sensors, embed_dim)
        self.hier_embeddings = nn.ModuleList(
            [
                nn.Embedding(np.max(X_hier[:, i]) + 1, embed_dim)
                for i in range(X_hier.shape[1])
            ]
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x_sensors, x_hier):
        x = self.tokenizer(x_sensors).unsqueeze(1)
        for i in range(x_hier.shape[1]):
            x_hier[:, i] = torch.clamp(
                x_hier[:, i], 0, self.hier_embeddings[i].num_embeddings - 1
            )
        hier_embeds = torch.stack(
            [emb(x_hier[:, i]) for i, emb in enumerate(self.hier_embeddings)], dim=1
        )
        x = torch.cat([x, hier_embeds], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)


# ✅ K-Fold Training
kf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
fold_results = []

X_sensors_tensor = torch.tensor(X_sensors_resampled, dtype=torch.float32)
X_hier_tensor = torch.tensor(X_hier_resampled, dtype=torch.long)
y_tensor = torch.tensor(y_resampled, dtype=torch.long)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_sensors_tensor, y_tensor)):
    print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")
    train_loader = DataLoader(
        SensorDataset(
            X_sensors_tensor[train_idx], X_hier_tensor[train_idx], y_tensor[train_idx]
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SensorDataset(
            X_sensors_tensor[val_idx], X_hier_tensor[val_idx], y_tensor[val_idx]
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = HierarchicalTransformerWithMHSA(
        num_sensors=X_sensors.shape[1],
        num_classes=len(np.unique(y_tensor)),
        num_heads=args.num_heads,
        depth=args.depth,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_val_loss = float("inf")
    patience, epochs_no_improve = args.patience, 0

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for sensors, hier, labels in train_loader:
            sensors, hier, labels = (
                sensors.to(device),
                hier.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            outputs = model(sensors, hier)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        val_loss, val_correct, val_total = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for sensors, hier, labels in val_loader:
                sensors, hier, labels = (
                    sensors.to(device),
                    hier.to(device),
                    labels.to(device),
                )
                outputs = model(sensors, hier)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        train_acc = correct / total

        print(
            f"Epoch {epoch + 1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("⏹ Early stopping")
                break

    # ✅ Evaluation
    model.load_state_dict(best_model_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sensors, hier, labels in val_loader:
            sensors, hier = sensors.to(device), hier.to(device)
            outputs = model(sensors, hier)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    kappa = cohen_kappa_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    fold_results.append((acc, prec, rec, f1, kappa, mcc))

# ✅ Summary
results = np.array(fold_results)
print("\n=== Final Cross-Validation Results ===")
print(f"✅ Accuracy     : {results[:, 0].mean():.4f}")
print(f"✅ Precision    : {results[:, 1].mean():.4f}")
print(f"✅ Recall       : {results[:, 2].mean():.4f}")
print(f"✅ F1-Score     : {results[:, 3].mean():.4f}")
print(f"✅ Cohen Kappa  : {results[:, 4].mean():.4f}")
print(f"✅ MCC          : {results[:, 5].mean():.4f}")
