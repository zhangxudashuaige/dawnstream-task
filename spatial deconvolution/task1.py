import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import argparse
import warnings
warnings.filterwarnings("ignore")

CELL_TYPES = [
    "Monocyte", "LAM", "KC", "T_cell", "Effector_T", "B_cell",
    "Neutrophil", "NK", "cDC", "pDC", "ILC1", "Fibroblast",
    "HSC", "VSMC", "Mesothelial_cell", "PC_LVEC", "PP_LVEC",
    "LSEC", "LyEC", "Cholangiocyte", "Hep", "LPLC"
]

class MLPRegressor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=22, dropout=0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.head(x)

def load_data(emb_path, data_path):
    adata = sc.read_h5ad(data_path)
    embeddings = np.load(emb_path)
    proportions = adata.obs[CELL_TYPES].values.astype(np.float32)
    return embeddings, proportions

def train_one_epoch(model, dl, opt, crit, device):
    model.train()
    total_loss, total = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / total

@torch.no_grad()
def evaluate(model, dl, crit, device, verbose=False):
    model.eval()
    total_loss, total = 0.0, 0
    all_pred, all_true = [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += crit(pred, y).item() * x.size(0)
        total += x.size(0)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())
    all_pred = np.concatenate(all_pred, 0)
    all_true = np.concatenate(all_true, 0)

    per_type_pcc = []
    for i in range(all_true.shape[1]):
        gt, pr = all_true[:, i], all_pred[:, i]
        if gt.std() < 1e-8:
            continue
        per_type_pcc.append(pearsonr(gt, pr)[0])
    mean_pcc = np.mean(per_type_pcc) if per_type_pcc else 0.0
    rmse = np.sqrt(((all_true - all_pred) ** 2).mean())

    if verbose:
        for i, ct in enumerate(CELL_TYPES):
            if all_true[:, i].std() > 1e-8:
                print(f"    {ct:20s}: PCC={pearsonr(all_true[:,i], all_pred[:,i])[0]:.4f}")

    return total_loss / total, mean_pcc, rmse

def run_single(emb_path, data_path, args):
    print(f"  加载 {emb_path}")
    emb, prop = load_data(emb_path, data_path)
    print(f"  Score范围: [{prop.min():.4f}, {prop.max():.4f}], 均值: {prop.mean():.4f}")

    X = torch.tensor(emb, dtype=torch.float32)
    y = torch.tensor(prop, dtype=torch.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    tr_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    te_dl = DataLoader(TensorDataset(X_te, y_te), batch_size=args.batch_size, shuffle=False)

    model = MLPRegressor(256, args.hidden_dim, len(CELL_TYPES), args.dropout).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.MSELoss()
    best_pcc, best_state = -1, None
    patience = 0

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, tr_dl, opt, crit, args.device)
        sch.step()
        _, val_pcc, val_rmse = evaluate(model, te_dl, crit, args.device)
        if val_pcc > best_pcc:
            best_pcc, best_state = val_pcc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if ep % 40 == 0 or ep == 1:
            print(f"    Ep {ep:3d} | Loss: {tr_loss:.4f} | PCC: {val_pcc:.4f} RMSE: {val_rmse:.4f}")
        if patience >= args.patience:
            print(f"    Early stop at {ep}")
            break

    model.load_state_dict(best_state)
    _, final_pcc, final_rmse = evaluate(model, te_dl, crit, args.device, verbose=True)
    return final_pcc, final_rmse

def run_cross(all_paths, samples, args):
    results = []
    for fold in range(len(samples)):
        test_s = samples[fold]
        train_paths = [all_paths[i] for i in range(len(samples)) if i != fold]
        train_names = [samples[i] for i in range(len(samples)) if i != fold]
        print(f"\nFold {fold+1}: 训 {train_names} -> 测 {test_s}")

        X_list, y_list = [], []
        for ep, dp in train_paths:
            e, p = load_data(ep, dp)
            X_list.append(torch.tensor(e, dtype=torch.float32))
            y_list.append(torch.tensor(p, dtype=torch.float32))
        X_all = torch.cat(X_list, 0)
        y_all = torch.cat(y_list, 0)
        X_tr, X_va, y_tr, y_va = train_test_split(X_all, y_all, test_size=0.1, random_state=args.seed)
        tr_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
        va_dl = DataLoader(TensorDataset(X_va, y_va), batch_size=args.batch_size, shuffle=False)

        model = MLPRegressor(256, args.hidden_dim, len(CELL_TYPES), args.dropout).to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        crit = nn.MSELoss()
        best_pcc, best_state = -1, None
        patience = 0
        for ep in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, tr_dl, opt, crit, args.device)
            _, va_pcc, _ = evaluate(model, va_dl, crit, args.device)
            if va_pcc > best_pcc:
                best_pcc, best_state = va_pcc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if ep % 40 == 0 or ep == 1:
                print(f"    Ep {ep:3d} | PCC: {va_pcc:.4f}")
            if patience >= args.patience:
                print(f"    Early stop at {ep}")
                break
        model.load_state_dict(best_state)

        te_emb, te_prop = load_data(all_paths[fold][0], all_paths[fold][1])
        Xt = torch.tensor(te_emb, dtype=torch.float32)
        yt = torch.tensor(te_prop, dtype=torch.float32)
        te_dl = DataLoader(TensorDataset(Xt, yt), batch_size=args.batch_size, shuffle=False)
        _, pcc, rmse = evaluate(model, te_dl, crit, args.device)
        results.append((test_s, pcc, rmse))
        print(f"  -> PCC: {pcc:.4f}, RMSE: {rmse:.4f}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="single", choices=["single", "cross"])
    parser.add_argument("--data_dir", type=str, default="work/test6/stofm_input")
    parser.add_argument("--samples", type=str, default="D0_DT2,D0_DY1,D17_FR4")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    samples = [s.strip() for s in args.samples.split(",")]
    all_paths = [(f"{args.data_dir}/{s}/ce_emb.npy", f"{args.data_dir}/{s}/data.h5ad") for s in samples]
    print(f"空间解卷积 | 模式: {args.mode} | 样本: {samples}")

    if args.mode == "single":
        print(f"\n{'='*50}\n每个样本单独跑 (80/20)\n{'='*50}")
        res = []
        for i, (ep, dp) in enumerate(all_paths):
            print(f"\n[{samples[i]}]")
            pcc, rmse = run_single(ep, dp, args)
            res.append((samples[i], pcc, rmse))
            print(f"  => PCC: {pcc:.4f}, RMSE: {rmse:.4f}")
    else:
        print(f"\n{'='*50}\n留一法交叉验证\n{'='*50}")
        res = run_cross(all_paths, samples, args)

    pccs, rmses = zip(*[(r[1], r[2]) for r in res])
    print(f"\n{'='*40}\n汇总")
    for s, p, r in res:
        print(f"  {s}: PCC={p:.4f}, RMSE={r:.4f}")
    print(f"  平均: PCC={np.mean(pccs):.4f}, RMSE={np.mean(rmses):.4f}")

if __name__ == "__main__":
    main()