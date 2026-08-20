"""Per-stratum full-vs-atac gap trajectory across evaluated epochs.

Single-checkpoint dose-response forests (plot_e1e10.py, e10) proved fragile:
the paired gap swings epoch to epoch on the same order as the effect (e8
flipped every stratum sign-negative before e9/e10 recovered).  This renders
the gap per DTU stratum at every epoch with per-gene TSVs, over the thin
all-gene history line (available for every epoch), so volatility is visible
instead of hidden behind one epoch's confidence intervals.
"""
import csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path

dtu = pd.read_csv("outputs/analysis/dtu_recomputed.tsv", sep="\t")[["gene_id","score_train"]]
rng = np.random.default_rng(1103); NB = 4000

def load(cond, ep):
    d = pd.read_csv(f"outputs/analysis/macro_eval/per_gene_{cond}_e{ep}.tsv", sep="\t")
    d = d[d.nll_molecule_mass > 0].copy()
    d["nll"] = d.nll_weighted_sum / d.nll_molecule_mass
    return d

def boot(x, nb=NB, batch=500):
    n=len(x); out=[]
    for s in range(0,nb,batch):
        k=min(batch,nb-s); out.append(x[rng.integers(0,n,size=(k,n))].mean(axis=1))
    return np.percentile(np.concatenate(out),[2.5,97.5])

epochs = sorted(
    int(p.stem.split("_e")[1])
    for p in Path("outputs/analysis/macro_eval").glob("per_gene_full_e*.tsv")
    if Path(f"outputs/analysis/macro_eval/per_gene_atac_e{p.stem.split('_e')[1]}.tsv").exists()
)

STRATA = [
    ("all",     "#333333", lambda s: s >= 0),
    ("DTU=0",   "#999999", lambda s: s == 0),
    (">0.65",   "#1a7f37", lambda s: s > 0.65),
    ("top-10%", "#7a4fbf", None),  # quantile computed per merge below
]

series = {name: {"mu": [], "lo": [], "hi": []} for name, _, _ in STRATA}
for ep in epochs:
    f, a = load("full", ep), load("atac", ep)
    g = f.merge(a, on="gene_id", suffixes=("_f","_a")).merge(dtu, on="gene_id").dropna(subset=["score_train"])
    g["delta"] = g.nll_a - g.nll_f
    q90 = g.score_train.quantile(0.9)
    for name, _, fn in STRATA:
        mask = g.score_train >= q90 if fn is None else fn(g.score_train)
        s = g[mask]; ci = boot(s.delta.values)
        series[name]["mu"].append(s.delta.mean())
        series[name]["lo"].append(ci[0]); series[name]["hi"].append(ci[1])

history_gap = {}
for cond in ("full", "atac"):
    with open(f"runs/fabric_v2_{cond}_macro_seed1103/history.tsv") as h:
        history_gap[cond] = {
            int(r["epoch"]): float(r["validation_compatible_path_nll_gene_macro"])
            for r in csv.DictReader(h, delimiter="\t")
        }
hist_epochs = sorted(set(history_gap["full"]) & set(history_gap["atac"]))
hist_gap = [history_gap["atac"][e] - history_gap["full"][e] for e in hist_epochs]

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(hist_epochs, hist_gap, "-", color="#bbbbbb", lw=1.2, zorder=1,
        label="all genes (history, every epoch)")
for i, (name, color, _) in enumerate(STRATA):
    x = np.array(epochs) + (i - 1.5) * 0.06
    mu = np.array(series[name]["mu"])
    lo, hi = np.array(series[name]["lo"]), np.array(series[name]["hi"])
    ax.errorbar(x, mu, yerr=[mu - lo, hi - mu], fmt="o-", color=color, lw=1.6,
                ms=5, capsize=3, label=name, zorder=3)
ax.axhline(0, color="black", lw=0.9)
full_lr_drop = 8.5  # full halves LR after e8; atac stays at base LR through e10
ax.axvline(full_lr_drop, color="#8c1c2b", ls="--", lw=1.1)
ax.annotate("full: LR/2\n(atac unchanged)", (full_lr_drop, 0.98),
            xycoords=("data", "axes fraction"), fontsize=8, color="#8c1c2b",
            ha="left", textcoords="offset points", xytext=(4, -16))
ax.set_xticks(hist_epochs)
ax.set_xlabel("epoch")
ax.set_ylabel("atac − full  per-gene macro NLL  (pos = RNA channel helps)")
ax.set_title("Gap trajectory by DTU stratum — gene-bootstrap 95% CIs\n"
             "(matched LR through e8; single-epoch forests hide this volatility)",
             fontsize=11)
ax.legend(fontsize=9, loc="lower left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("outputs/analysis/gap_trajectory.png", dpi=150)
print("outputs/analysis/gap_trajectory.png")

for name, _, _ in STRATA:
    row = "  ".join(
        f"e{ep}:{mu:+.4f}" for ep, mu in zip(epochs, series[name]["mu"])
    )
    print(f"{name:>8}  {row}")
