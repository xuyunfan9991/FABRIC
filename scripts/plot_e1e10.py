"""Consolidated e1/e10 stratified report for the gene_macro runs."""
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

STRATA = [("DTU=0", lambda s: s==0), ("0–0.1", lambda s:(s>0)&(s<=0.1)),
          ("0.1–0.35", lambda s:(s>0.1)&(s<=0.35)), ("0.35–0.65", lambda s:(s>0.35)&(s<=0.65)),
          (">0.65", lambda s:s>0.65)]
COLOR = {"full": "#1b6ca8", "atac": "#d1495b"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

# P1: improvement by stratum
ax = axes[0]
x = np.arange(len(STRATA))
for i, cond in enumerate(("full","atac")):
    a,b = load(cond,1), load(cond,10)
    m = a.merge(b,on="gene_id",suffixes=("_1","_10")).merge(dtu,on="gene_id").dropna(subset=["DTU_score"])
    m["imp"]=m.nll_1-m.nll_10
    vals, err = [], []
    for _, f in STRATA:
        g=m[f(m.DTU_score)]; ci=boot(g.imp.values)
        vals.append(g.imp.mean()); err.append([g.imp.mean()-ci[0], ci[1]-g.imp.mean()])
    ax.bar(x+(i-0.5)*0.38, vals, 0.38, yerr=np.array(err).T, capsize=3,
           color=COLOR[cond], label=f"{cond}_macro")
ax.axhline(0,color="black",lw=0.9)
ax.axhline(-0.00856, color="#8c1c2b", ls=":", lw=1.4)
ax.annotate("old objective, high-DTU late window: −0.009 (regressed)",
            (0.02,0.055), xycoords="axes fraction", fontsize=8, color="#8c1c2b")
ax.set_xticks(x); ax.set_xticklabels([n for n,_ in STRATA], fontsize=9)
ax.set_xlabel("frozen DTU prior stratum"); ax.set_ylabel("NLL improvement e1→e10")
ax.set_title("All strata now learn — incl. high-DTU", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")

# P2: dose-response at e10
ax = axes[1]
f10,a10 = load("full",10), load("atac",10)
g = f10.merge(a10,on="gene_id",suffixes=("_f","_a")).merge(dtu,on="gene_id").dropna(subset=["DTU_score"])
g["delta"]=g.nll_a-g.nll_f
names = ["all","DTU=0","0–0.1","0.1–0.35","0.35–0.65",">0.65","top-10%"]
masks = [g.DTU_score>=0, g.DTU_score==0, (g.DTU_score>0)&(g.DTU_score<=0.1),
         (g.DTU_score>0.1)&(g.DTU_score<=0.35), (g.DTU_score>0.35)&(g.DTU_score<=0.65),
         g.DTU_score>0.65, g.DTU_score>=g.DTU_score.quantile(0.9)]
for i,(n,msk) in enumerate(zip(names,masks)):
    s=g[msk]; ci=boot(s.delta.values); mu=s.delta.mean()
    sig = ci[0]>0 or ci[1]<0
    ax.errorbar(mu, len(names)-1-i, xerr=[[mu-ci[0]],[ci[1]-mu]], fmt="o",
                color="#1a7f37" if sig else "#999999", ms=8 if sig else 6,
                capsize=4, lw=2)
ax.axvline(0,color="black",lw=0.9)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names[::-1], fontsize=9)
ax.set_xlabel("ATAC − Full NLL at e10  (positive = RNA channel helps)")
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.set_title("Dose-response: gain concentrates in high-DTU\n(green = 95% CI excludes 0)", fontsize=10.5)
ax.grid(alpha=0.25, axis="x")

# P3: mass-quintile improvement (overfit check)
ax = axes[2]
a,b = load("full",1), load("full",10)
m = a.merge(b,on="gene_id",suffixes=("_1","_10")); m["imp"]=m.nll_1-m.nll_10
m["q"]=pd.qcut(m.nll_molecule_mass_1,5,labels=False)
vals,errs,labs=[],[],[]
for q,gg in m.groupby("q"):
    ci=boot(gg.imp.values); vals.append(gg.imp.mean())
    errs.append([gg.imp.mean()-ci[0],ci[1]-gg.imp.mean()])
    labs.append(f"{gg.nll_molecule_mass_1.min():.0f}–{gg.nll_molecule_mass_1.max():.0f}")
ax.bar(range(5), vals, 0.6, yerr=np.array(errs).T, capsize=3, color="#4a7c59")
ax.axhline(0,color="black",lw=0.9)
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=8, rotation=20)
ax.set_xlabel("gene validation molecule mass (quintile)")
ax.set_ylabel("NLL improvement e1→e10 (full_macro)")
ax.set_title("No small-gene overfitting: every mass tier improves", fontsize=10.5)
ax.grid(alpha=0.25, axis="y")

fig.suptitle("gene_macro e1 vs e10 — stratified verdicts (1 seed; e10 gap carries LR-schedule caveat)", fontsize=12)
fig.tight_layout()
fig.savefig("outputs/analysis/e1e10_report.png", dpi=150)
print("outputs/analysis/e1e10_report.png")
