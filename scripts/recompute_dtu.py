"""Recompute the DTU JS score, replicating data/DTU_score.R, in two scopes:
- all cells   (fidelity check against the original DTU_result_sorted.xlsx)
- train cells (the leakage-free version the plan requires)
"""
import numpy as np, pandas as pd, scipy.sparse as sp, re, time

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

ROOT = "/home2/xyf/project/FABRIC"
MTX  = "/home2/xyf/project/PRISM/data/matrix/tx_matrix_ONT/tx_matrix_ONT.mtx"
TXS  = "/home2/xyf/project/PRISM/data/matrix/tx_matrix_ONT/transcripts.tsv"
GTF  = "/home2/xyf/project/PRISM/data/gtf/transcripts_filtered.gtf"
META = "/home1/xyf/project/Multi_Omic/data/gene_matrix_Illumina/cell_meta_final.aligned.tsv"
MCI  = f"{ROOT}/data/processed/fabric_ont_gene_selection_v3/matrix_cell_index.parquet"

log("loading matrix (140M nnz)...")
df = pd.read_csv(MTX, sep=" ", skiprows=2, names=["r","c","v"],
                 dtype={"r":np.int32,"c":np.int32,"v":np.int32})
X = sp.csr_matrix((df.v.values, (df.r.values-1, df.c.values-1)), shape=(101067, 217933))
del df
log(f"matrix ready: {X.shape}, nnz={X.nnz:,}")

cw = pd.read_parquet(f"{ROOT}/data/processed/fabric_ont_gene_selection_v3/transcript_crosswalk_audit.parquet")
cw = cw.sort_values("matrix_row_0based")
assert (cw.matrix_row_0based.values == np.arange(len(cw))).all()
gene_of = cw.gene_id.fillna("").values.astype(str)
log(f"tx->gene mapped via crosswalk: {(gene_of != '').sum()}/{len(gene_of)}")

mci = pd.read_parquet(MCI)
meta = pd.read_csv(META, sep="\t", usecols=["cell_id","cell_type"])
meta = meta.set_index("cell_id").cell_type
ct_of_col = meta.reindex(mci.sort_values("matrix_column_0based").matrix_barcode).values
split_of_col = mci.sort_values("matrix_column_0based").split.values
log(f"cells annotated: {pd.notna(ct_of_col).sum():,}/{len(ct_of_col):,}; "
    f"types={pd.Series(ct_of_col).nunique()}")

def js_vs_uniform(x):
    x = x[~np.isnan(x)]
    if len(x) < 2: return np.nan
    s = x.sum()
    if s == 0: return 0.0
    p = x/s; n = len(p); q = np.full(n, 1.0/n); m = 0.5*(p+q)
    def kl(a,b):
        idx = (a>0)&(b>0)
        return float(np.sum(a[idx]*np.log2(a[idx]/b[idx]))) if idx.any() else 0.0
    return max(0.5*kl(p,m)+0.5*kl(q,m), 0.0)

def score_scope(mask_cols, tag):
    cols = np.flatnonzero(mask_cols & pd.notna(ct_of_col))
    cts  = pd.Series(ct_of_col[cols])
    codes, uniq = pd.factorize(cts)
    H = sp.csr_matrix((np.ones(len(cols)), (cols, codes)), shape=(X.shape[1], len(uniq)))
    A = np.asarray((X @ H).todense())          # tx x celltype pooled counts
    log(f"[{tag}] pooled: {A.shape}, celltypes={len(uniq)}, cells={len(cols):,}")

    rows = []
    order = np.argsort(gene_of, kind="stable")
    bounds = np.flatnonzero(np.r_[True, gene_of[order][1:] != gene_of[order][:-1], True])
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        idx = order[b0:b1]; gid = gene_of[idx[0]]
        if not gid or len(idx) < 2:            # min_transcripts = 2
            continue
        G = A[idx]                              # tx x ct
        gsum = G.sum(axis=0)
        valid = gsum > 0
        if valid.sum() < 2:                     # min_celltypes = 2
            rows.append((gid, len(idx), int(valid.sum()), np.nan, 0)); continue
        P = G[:, valid] / gsum[valid]           # PSI
        mx = P.max(axis=0)
        dom_share = np.zeros_like(P)
        for j in range(P.shape[1]):
            w = np.abs(P[:, j] - mx[j]) < 1e-9
            dom_share[w, j] = 1.0/ w.sum()
        dom_counts = dom_share.sum(axis=1)
        qual = np.flatnonzero(dom_counts >= 1)
        if len(qual) <= 1:
            rows.append((gid, len(idx), int(valid.sum()), 0.0, len(qual))); continue
        props = dom_counts[qual]/dom_counts[qual].sum()
        js = np.array([js_vs_uniform(P[t]) for t in qual])
        js = np.nan_to_num(js)
        score = float(-np.sum(js*props*np.log2(props)))
        rows.append((gid, len(idx), int(valid.sum()), score, len(qual)))
    out = pd.DataFrame(rows, columns=["gene_id","n_tx","n_valid_ct",f"score_{tag}",f"n_dom_{tag}"])
    log(f"[{tag}] scored {len(out)} genes")
    return out

all_scores   = score_scope(np.ones(len(ct_of_col), dtype=bool), "all")
train_scores = score_scope(split_of_col == "train", "train")

merged = all_scores.merge(train_scores[["gene_id","score_train","n_dom_train"]], on="gene_id", how="outer")
merged.to_csv("outputs/analysis/dtu_recomputed.tsv", sep="\t", index=False)
log(f"written outputs/analysis/dtu_recomputed.tsv ({len(merged)} genes)")
