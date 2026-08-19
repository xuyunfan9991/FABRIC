"""Forest plot: macro vs molecule-weighted gap with gene-level bootstrap CIs."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

dtu = pd.read_excel("data/DTU_result_sorted.xlsx")[["gene_id","DTU_score","top_DTU_gene"]]
rng = np.random.default_rng(1103); NB = 4000

def boot(delta, contrib, mass, batch=250):
    n=len(delta); mac=[]; wei=[]
    for s in range(0, NB, batch):
        k=min(batch, NB-s); idx=rng.integers(0,n,size=(k,n))
        mac.append(delta[idx].mean(axis=1))
        wei.append(contrib[idx].sum(axis=1)/mass[idx].sum(axis=1))
    return np.concatenate(mac), np.concatenate(wei)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
labels = ["All genes", "top_DTU = yes", "top_DTU = no", "DTU = 0", "DTU > 0.6"]

for ax, ep in zip(axes, (19, 30)):
    f = pd.read_csv(f"outputs/analysis/per_gene_full_e{ep}.tsv", sep="\t")
    a = pd.read_csv(f"outputs/analysis/per_gene_atac_e{ep}.tsv", sep="\t")
    m = f[f.nll_molecule_mass>0].merge(a,on="gene_id",suffixes=("_f","_a")).merge(dtu,on="gene_id")
    m["delta"]=m.nll_weighted_sum_a/m.nll_molecule_mass_a - m.nll_weighted_sum_f/m.nll_molecule_mass_f
    m["contrib"]=m.nll_weighted_sum_a-m.nll_weighted_sum_f
    groups=[m, m[m.top_DTU_gene=="yes"], m[m.top_DTU_gene=="no"],
            m[m.DTU_score==0], m[m.DTU_score>0.6]]
    for i,(lab,g) in enumerate(zip(labels, groups)):
        mac, wei = boot(g.delta.values, g.contrib.values, g.nll_molecule_mass_f.values)
        for off, vals, point, col, name in (
            (+0.16, mac, g.delta.mean(), "#4a7c59", "macro"),
            (-0.16, wei, g.contrib.sum()/g.nll_molecule_mass_f.sum(), "#e08a3c", "molecule-weighted")):
            lo, hi = np.percentile(vals,[2.5,97.5])
            y = len(labels)-1-i+off
            sig = lo > 0 or hi < 0
            ax.plot([lo,hi],[y,y], color=col, lw=2.4, solid_capstyle="butt", alpha=1 if sig else 0.45)
            ax.plot(point, y, "o" if sig else "o", ms=8 if sig else 6, color=col,
                    markeredgecolor="black" if sig else "none", markeredgewidth=1.1,
                    label=name if i==0 else None, zorder=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("NLL gap: atac − full   (positive = full better)")
    ax.set_title(f"epoch {ep}", fontsize=12)
    ax.grid(alpha=0.25, axis="x")
axes[0].legend(loc="lower right", fontsize=9)
fig.suptitle("Gene-level bootstrap (4000×): filled marker = 95% CI excludes zero", fontsize=12.5)
fig.tight_layout()
fig.savefig("outputs/analysis/forest.png", dpi=150)
print("outputs/analysis/forest.png")
