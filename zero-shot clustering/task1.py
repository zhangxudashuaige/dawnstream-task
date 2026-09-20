import scanpy as sc
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score
import argparse
import warnings
warnings.filterwarnings("ignore")

def load_data(emb_path, data_path, label_key):
    adata = sc.read_h5ad(data_path)
    embeddings = np.load(emb_path)
    labels = adata.obs[label_key].values
    return embeddings, labels

def cluster_and_eval(embeddings, labels_str):
    le = LabelEncoder()
    labels = le.fit_transform(labels_str)
    n_clusters = len(le.classes_)

    print(f"  细胞数: {len(embeddings)}, 真实类别数: {n_clusters}")
    print(f"  类别分布: {dict(zip(le.classes_, np.bincount(labels)))}")

    # KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_pred = kmeans.fit_predict(embeddings)

    ari = adjusted_rand_score(labels, cluster_pred)
    nmi = normalized_mutual_info_score(labels, cluster_pred)
    homo = homogeneity_score(labels, cluster_pred)

    return ari, nmi, homo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="work/test5/stofm_input")
    parser.add_argument("--samples", type=str, default="1_0,4_0")
    parser.add_argument("--label_key", type=str, default="cell_type")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    samples = [s.strip() for s in args.samples.split(",")]

    print(f"零样本聚类 | 标签: {args.label_key}")
    results = []
    for s in samples:
        ep = f"{args.data_dir}/{s}/stofm_emb.npy"
        dp = f"{args.data_dir}/{s}/data.h5ad"
        print(f"\n[{s}]")
        emb, labs = load_data(ep, dp, args.label_key)
        ari, nmi, homo = cluster_and_eval(emb, labs)
        results.append((s, ari, nmi, homo))
        print(f"  ARI: {ari:.4f}, NMI: {nmi:.4f}, Homo: {homo:.4f}")

    print(f"\n{'='*40}\n汇总")
    aris, nmis, homos = zip(*[(r[1], r[2], r[3]) for r in results])
    for s, a, n, h in results:
        print(f"  {s}: ARI={a:.4f}, NMI={n:.4f}, Homo={h:.4f}")
    print(f"  平均: ARI={np.mean(aris):.4f}, NMI={np.mean(nmis):.4f}")

if __name__ == "__main__":
    main()
