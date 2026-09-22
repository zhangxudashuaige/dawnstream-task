# -*- coding: utf-8 -*-
"""
下游任务训练：h5ad 从 Hugging Face 下载，embedding 从本地读取，做 MLP 分类。

数据来源：
    - h5ad      : Hugging Face 数据集仓库 "anuosi666/tissue-region-segmentation"（已上传）
    - embedding : 本地目录，结构为 {emb_dir}/{样本名}/ce_emb.npy（不上传 HF）

运行前安装依赖：
    pip install scanpy anndata torch scikit-learn huggingface_hub

用法示例（--emb_dir 指向本地 embedding 根目录）：
    python train_downstream.py --emb_dir "D:/mydata/emb" --label_key celltype
    python train_downstream.py --emb_dir "D:/mydata/emb" --mode cross --label_key celltype
"""
import os
import argparse
import warnings
from collections import Counter

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# ============ Hugging Face 配置 ============
HF_REPO = "anuosi666/tissue-region-segmentation"   # 你的数据集仓库


def get_h5ad_file(sample: str) -> str:
    """样本名 -> HF 仓库里的 h5ad 相对路径。"""
    key = sample.upper()                                  # e2s3 -> E2S3
    return f"CS12-13/CS12-13_{key}_HESTA.h5ad"


def get_emb_file(sample: str, emb_dir: str) -> str:
    """样本名 -> 本地 embedding 路径（embedding 不上传，直接在本地读）。"""
    return os.path.join(emb_dir, sample, "ce_emb.npy")


def download(filename: str) -> str:
    """从 HF 下载文件到本地缓存，返回本地路径（重复调用不重复下载）。"""
    return hf_hub_download(repo_id=HF_REPO, filename=filename, repo_type="dataset")


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.head(x)


def load_data(emb_path, h5ad_file, label_key):
    adata = sc.read_h5ad(download(h5ad_file))   # h5ad 从 HF 下载
    embeddings = np.load(emb_path)              # embedding 从本地读
    labels = adata.obs[label_key].values
    return embeddings, labels


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")
    ari = adjusted_rand_score(all_labels, all_preds)
    return total_loss / total, acc, f1, ari


def train_model(model, train_loader, val_loader, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for ep in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, crit, device)
        val_loss, val_acc, val_f1, val_ari = evaluate(model, val_loader, crit, device)
        sch.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if ep % 40 == 0 or ep == 1:
            print(f"    Ep {ep:3d} | Tr acc: {train_acc:.4f} | Val acc: {val_acc:.4f} f1: {val_f1:.4f}")

        if patience_counter >= args.patience:
            print(f"    Early stop at epoch {ep}")
            break

    model.load_state_dict(best_state)


def run_single(emb_path, data_path, args):
    print(f"  加载 embedding: {emb_path}")
    emb, labs_str = load_data(emb_path, data_path, args.label_key)
    le = LabelEncoder().fit(labs_str)
    nc = len(le.classes_)
    print(f"  细胞: {len(emb)}, 类别: {nc}")

    X = torch.tensor(emb, dtype=torch.float32)
    y = torch.tensor(le.transform(labs_str), dtype=torch.long)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    te_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=args.batch_size, shuffle=False)

    model = MLPClassifier(256, args.hidden_dim, nc, args.dropout).to(args.device)
    train_model(model, tr_loader, te_loader, args, args.device)
    _, acc, f1, ari = evaluate(model, te_loader, nn.CrossEntropyLoss(), args.device)
    return acc, f1, ari


def run_cross(all_paths, samples, args, global_le, nc):
    crit = nn.CrossEntropyLoss()
    results = []

    for fold in range(len(samples)):
        test_s = samples[fold]
        train_paths = [all_paths[i] for i in range(len(samples)) if i != fold]
        train_names = [samples[i] for i in range(len(samples)) if i != fold]
        print(f"\nFold {fold+1}: 训 {train_names} -> 测 {test_s}")

        X_list, y_list = [], []
        for ep, dp in train_paths:
            e, l = load_data(ep, dp, args.label_key)
            X_list.append(torch.tensor(e, dtype=torch.float32))
            y_list.append(torch.tensor(global_le.transform(l), dtype=torch.long))
        X_all = torch.cat(X_list, 0)
        y_all = torch.cat(y_list, 0)

        X_tr, X_va, y_tr, y_va = train_test_split(
            X_all, y_all, test_size=0.1, stratify=y_all, random_state=args.seed
        )
        tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
        va_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=args.batch_size, shuffle=False)

        model = MLPClassifier(256, args.hidden_dim, nc, args.dropout).to(args.device)
        train_model(model, tr_loader, va_loader, args, args.device)

        te, tl_str = load_data(all_paths[fold][0], all_paths[fold][1], args.label_key)
        Xt = torch.tensor(te, dtype=torch.float32)
        yt = torch.tensor(global_le.transform(tl_str), dtype=torch.long)
        te_loader = DataLoader(TensorDataset(Xt, yt), batch_size=args.batch_size, shuffle=False)
        _, acc, f1, ari = evaluate(model, te_loader, crit, args.device)
        results.append((test_s, acc, f1, ari))
        print(f"  -> Acc: {acc:.4f}, F1: {f1:.4f}, ARI: {ari:.4f}")

    return results


def main():
    global HF_REPO
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="single", choices=["single", "cross"],
                        help="single=每个样本内部划分, cross=留一法交叉验证")
    parser.add_argument("--repo_id", type=str, default=HF_REPO,
                        help="Hugging Face 数据集仓库 id（存 h5ad）")
    parser.add_argument("--emb_dir", type=str, required=True,
                        help="本地 embedding 根目录，结构为 {emb_dir}/{样本名}/ce_emb.npy")
    parser.add_argument("--samples", type=str, default="e2s3,e2s4,e2s5,e2s6")
    parser.add_argument("--label_key", type=str, default="celltype")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    HF_REPO = args.repo_id

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    samples = [s.strip() for s in args.samples.split(",")]
    # (本地 emb 路径, HF 上的 h5ad 相对路径)
    all_paths = [(get_emb_file(s, args.emb_dir), get_h5ad_file(s)) for s in samples]
    print(f"模式: {args.mode}, 样本: {samples}")
    print(f"h5ad 仓库: {HF_REPO}, embedding 目录: {args.emb_dir}")

    if args.mode == "single":
        print(f"\n{'='*50}\n每个样本单独跑 ({1-args.test_size:.0%}/{args.test_size:.0%} 划分)\n{'='*50}")
        res = []
        for i, (ep, dp) in enumerate(all_paths):
            print(f"\n[{samples[i]}]")
            acc, f1, ari = run_single(ep, dp, args)
            res.append((samples[i], acc, f1, ari))
            print(f"  => Acc: {acc:.4f}, F1: {f1:.4f}, ARI: {ari:.4f}")
    else:
        all_labels = []
        for ep, dp in all_paths:
            _, labs = load_data(ep, dp, args.label_key)
            all_labels.extend(labs)
        le = LabelEncoder().fit(all_labels)
        nc = len(le.classes_)
        print(f"类别: {nc}, {list(le.classes_)}\n{'='*50}")
        res = run_cross(all_paths, samples, args, le, nc)

    accs, f1s, aris = zip(*[(r[1], r[2], r[3]) for r in res])
    print(f"\n{'='*50}\n汇总 ({args.mode})\n{'='*50}")
    for s, a, f, r in res:
        print(f"  {s}: Acc={a:.4f}, F1={f:.4f}, ARI={r:.4f}")
    print(f"  平均: Acc={np.mean(accs):.4f}, F1={np.mean(f1s):.4f}, ARI={np.mean(aris):.4f}")


if __name__ == "__main__":
    main()