import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score
import argparse
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

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

def load_data(emb_path, data_path, label_key):
    adata = sc.read_h5ad(data_path)
    embeddings = np.load(emb_path)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_path", type=str, default="work/test3/stofm_input/ce_emb.npy")
    parser.add_argument("--data_path", type=str, default="work/test3/stofm_input/data.h5ad")
    parser.add_argument("--label_key", type=str, default="annotation")
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"加载 {args.emb_path}")
    emb, labs_str = load_data(args.emb_path, args.data_path, args.label_key)
    le = LabelEncoder().fit(labs_str)
    nc = len(le.classes_)
    print(f"细胞数: {len(emb)}, 类别数: {nc}")
    print(f"类别: {list(le.classes_)}")
    print(f"分布: {dict(Counter(labs_str))}")

    X = torch.tensor(emb, dtype=torch.float32)
    y = torch.tensor(le.transform(labs_str), dtype=torch.long)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    te_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=args.batch_size, shuffle=False)

    model = MLPClassifier(256, args.hidden_dim, nc, args.dropout).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for ep in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, tr_loader, opt, crit, args.device)
        val_loss, val_acc, val_f1, val_ari = evaluate(model, te_loader, crit, args.device)
        sch.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if ep % 40 == 0 or ep == 1:
            print(f"Ep {ep:3d} | Tr acc: {train_acc:.4f} | Val acc: {val_acc:.4f} f1: {val_f1:.4f}")

        if patience_counter >= args.patience:
            print(f"Early stop at epoch {ep}")
            break

    model.load_state_dict(best_state)
    _, final_acc, final_f1, final_ari = evaluate(model, te_loader, crit, args.device)
    print(f"\n最终结果: Acc={final_acc:.4f}, F1={final_f1:.4f}, ARI={final_ari:.4f}")

    model.load_state_dict(best_state)
    all_loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=False)
    _, all_acc, all_f1, all_ari = evaluate(model, all_loader, crit, args.device)
    print(f"全数据集: Acc={all_acc:.4f}, F1={all_f1:.4f}, ARI={all_ari:.4f}")

if __name__ == "__main__":
    main()