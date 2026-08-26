"""Matched-LR dose-response check for the gene_macro paired runs.

The e10 dose-response (plot_e1e10.py) carries an LR-schedule caveat: the full
run's reduce_on_plateau halved its LR after e8 while the atac run was still at
the base LR, so the e10 full-atac gap mixes objective and schedule.  Both arms
share the base LR through e8, so e7/e8 are schedule-clean comparison points.
This renders the same DTU-stratified gap at those epochs.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

dtu = pd.read_excel("data/DTU_result_sorted.xlsx")[["gene_id", "DTU_score"]]
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

names = ["all","DTU=0","0–0.1","0.1–0.35","0.35–0.65",">0.65","top-10%"]
def strata(g):
    return [g.DTU_score>=0, g.DTU_score==0, (g.DTU_score>0)&(g.DTU_score<=0.1),
            (g.DTU_score>0.1)&(g.DTU_score<=0.35), (g.DTU_score>0.35)&(g.DTU_score<=0.65),
            g.DTU_score>0.65, g.DTU_score>=g.DTU_score.quantile(0.9)]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, sharex=True)
for ax, ep in zip(axes, (7, 8)):
    f, a = load("full", ep), load("atac", ep)
    g = f.merge(a,on="gene_id",suffixes=("_f","_a")).merge(dtu,on="gene_id").dropna(subset=["DTU_score"])
    g["delta"] = g.nll_a - g.nll_f
    for i,(name,mask) in enumerate(zip(names, strata(g))):
        s=g[mask]; ci=boot(s.delta.values); mu=s.delta.mean()
        sig = ci[0]>0 or ci[1]<0
        ax.errorbar(mu, len(names)-1-i, xerr=[[mu-ci[0]],[ci[1]-mu]], fmt="o",
                    color="#1a7f37" if sig else "#999999", ms=8 if sig else 6,
                    capsize=4, lw=2)
    ax.axvline(0,color="black",lw=0.9)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("ATAC − Full NLL  (positive = RNA channel helps)")
    ax.set_title(f"epoch {ep}", fontsize=11)
    ax.grid(alpha=0.25, axis="x")
fig.suptitle("Matched-LR dose-response (both arms at base LR 5e-5)\n"
             "green = gene-bootstrap 95% CI excludes 0", fontsize=11.5)
fig.tight_layout()
fig.savefig("outputs/analysis/matched_lr_dose_response.png", dpi=150)
print("outputs/analysis/matched_lr_dose_response.png")

for ep in (7, 8):
    f, a = load("full", ep), load("atac", ep)
    g = f.merge(a,on="gene_id",suffixes=("_f","_a")).merge(dtu,on="gene_id").dropna(subset=["DTU_score"])
    g["delta"] = g.nll_a - g.nll_f
    print(f"e{ep}:")
    for name, mask in zip(names, strata(g)):
        s=g[mask]; ci=boot(s.delta.values)
        sig = "SIG" if (ci[0]>0 or ci[1]<0) else "   "
        print(f"  {name:>9} n={len(s):>5}  {s.delta.mean():+.5f}  [{ci[0]:+.5f}, {ci[1]:+.5f}] {sig}")
