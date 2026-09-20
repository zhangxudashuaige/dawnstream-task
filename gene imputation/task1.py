"""Gene expression imputation: single MLP -> multiple target genes.
Train on section_1 (all cells), test on section_2.
Uses pre-computed stofm_emb.npy from stofm_input/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def load_section(data_dir, section):
    emb = np.load(f"{data_dir}/{section}/stofm_emb.npy").astype(np.float32)
    a = ad.read_h5ad(f"{data_dir}/{section}/data.h5ad")
    expr = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    return emb, expr.astype(np.float32), list(a.var_names)

def pearson_r(a, b, axis=None):
    a, b = a.astype(np.float64), b.astype(np.float64)
    a = a - a.mean(axis=axis, keepdims=True)
    b = b - b.mean(axis=axis, keepdims=True)
    den = np.sqrt(np.sum(a * a, axis=axis) * np.sum(b * b, axis=axis))
    return np.divide(np.sum(a * b, axis=axis), den,
                     out=np.full_like(den, np.nan), where=den > 0)

def compute_metrics(target, prediction):
    error = prediction.astype(np.float64) - target
    by_gene = pearson_r(target, prediction, axis=0)
    by_cell = pearson_r(target, prediction, axis=1)
    return {
        "pearson_flat": float(pearson_r(target, prediction)),
        "pearson_gene_mean": float(np.nanmean(by_gene)),
        "pearson_cell_mean": float(np.nanmean(by_cell)),
        "valid_genes": int(np.isfinite(by_gene).sum()),
        "valid_cells": int(np.isfinite(by_cell).sum()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
    }, by_gene

def train_one_seed(args, data_dir, seed):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)

    # section_1 = 训练, section_2 = 测试
    emb_tr, expr_tr, genes_tr = load_section(data_dir, "section_1")
    emb_te, expr_te, genes_te = load_section(data_dir, "section_2")
    assert genes_tr == genes_te, "Gene lists must match between sections"
    print(f"训练(s1): {emb_tr.shape[0]} | 测试(s2): {emb_te.shape[0]} | 基因: {len(genes_tr)}")

    # 随机选目标基因
    rng = np.random.default_rng(seed)
    n_genes = min(args.n_genes, expr_tr.shape[1])
    target_idx = np.sort(rng.choice(expr_tr.shape[1], n_genes, replace=False))
    print(f"目标基因: {n_genes}")

    Y_tr = expr_tr[:, target_idx]
    Y_te = expr_te[:, target_idx]

    # Z-score（训练集统计量）
    mean, std = emb_tr.mean(axis=0), emb_tr.std(axis=0)
    std = np.maximum(std, 1e-6)
    X_tr = (emb_tr - mean) / std
    X_te = (emb_te - mean) / std

    # 单 MLP → 所有目标基因
    net = nn.Sequential(
        nn.Linear(X_tr.shape[1], args.hidden[0]), nn.LeakyReLU(0.01),
        nn.Linear(args.hidden[0], args.hidden[1]), nn.LeakyReLU(0.01),
        nn.Linear(args.hidden[1], n_genes),
    ).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr)),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    def predict(arr):
        net.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(arr), args.batch_size):
                chunk = torch.from_numpy(arr[start:start + args.batch_size]).to(device)
                chunks.append(net(chunk).cpu().numpy())
        return np.concatenate(chunks)

    # 无 val split，跑满所有 epoch
    for epoch in range(1, args.epochs + 1):
        net.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  seed={seed} ep={epoch:3d} train_mse={total / len(X_tr):.6f}")

    # 测试 section_2
    pred = predict(X_te)
    result, per_gene = compute_metrics(Y_te, pred)
    result.update(seed=seed, epochs=args.epochs, target_genes=n_genes)

    # 保存
    out_dir = args.output / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_names = [genes_tr[i] for i in target_idx]
    pd.DataFrame({"gene": target_names, "pearson": per_gene}).to_csv(
        out_dir / "per_gene.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  flat_pcc={result['pearson_flat']:.4f} gene_pcc={result['pearson_gene_mean']:.4f} rmse={result['rmse']:.4f}")
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=str, default="work/test7/stofm_input")
    p.add_argument("--output", type=Path, default=ROOT / "work/test7/imputation_runs")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n_genes", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--hidden", type=int, nargs=2, default=[256, 128])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"基因表达插补 | section_1→section_2 | 目标基因: {args.n_genes} | seeds: {args.seeds}")

    results = []
    for seed in args.seeds:
        print(f"\n{'='*50}\nSeed {seed}\n{'='*50}")
        results.append(train_one_seed(args, args.data_dir, seed))

    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output / "results.csv", index=False)
        keys = ["pearson_flat", "pearson_gene_mean", "pearson_cell_mean", "rmse", "mae"]
        summary = {k: {"mean": float(np.mean([r[k] for r in results])),
                       "std": float(np.std([r[k] for r in results]))}
                   for k in keys if k in results[0]}
        with open(args.output / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    main()