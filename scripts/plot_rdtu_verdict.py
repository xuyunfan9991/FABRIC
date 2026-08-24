"""Verdict figure for R005 (variant 3) vs variant 2 on the DTU question.

Panel 1: high-DTU (>0.65) full-atac gap at every epoch with per-gene TSVs,
variant 3 solid vs variant 2 dotted, showing the seed-locked oscillation and
where each run stopped.  Panel 2: within-arm improvement of variant 3 over
variant 2 by DTU stratum at e7/e8 -- the direct effect of the new loss.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dtu = pd.read_csv("outputs/analysis/dtu_recomputed.tsv", sep="\t")[["gene_id","score_train"]]
rng = np.random.default_rng(1103)

def load(base, cond, ep):
    d = pd.read_csv(f"outputs/analysis/{base}/per_gene_{cond}_e{ep}.tsv", sep="\t")
    d = d[d.nll_molecule_mass > 0].copy()
    d["nll"] = d.nll_weighted_sum / d.nll_molecule_mass
    return d[["gene_id", "nll"]]

def boot(x, nb=4000, batch=500):
    n = len(x)
    out = []
    for s in range(0, nb, batch):
        k = min(batch, nb - s)
        out.append(x[rng.integers(0, n, size=(k, n))].mean(axis=1))
    return np.percentile(np.concatenate(out),[2.5,97.5])

def hi_gap(base, ep):
    f, a = load(base,"full",ep), load(base,"atac",ep)
    g = f.merge(a,on="gene_id",suffixes=("_f","_a")).merge(dtu,on="gene_id").dropna(subset=["score_train"])
    s = g[g.score_train > 0.65]
    d = (s.nll_a - s.nll_f).values
    ci = boot(d)
    return d.mean(), ci

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

ax = axes[0]
for base, eps, style, color, label in (
        ("rdtu_eval", (1,6,7,8), "o-", "#1a7f37", "variant 3 (new loss)"),
        ("macro_eval", (1,7,8,9,10), "s:", "#888888", "variant 2 (equal weight)")):
    mus, los, his_ = [], [], []
    for ep in eps:
        mu, ci = hi_gap(base, ep)
        mus.append(mu)
        los.append(mu-ci[0])
        his_.append(ci[1]-mu)
    ax.errorbar(eps, mus, yerr=[los,his_], fmt=style, color=color, lw=1.8,
                ms=6, capsize=4, label=label)
ax.axhline(0, color="black", lw=0.9)
ax.annotate("v3 stopped e8/e9\n(v2's e9-e10 recovered)", (8.15, -0.021),
            fontsize=8.5, color="#1a7f37")
ax.set_xlabel("epoch")
ax.set_ylabel("atac − full NLL, DTU>0.65 stratum")
ax.set_title("High-DTU gap: same seed-locked oscillation,\nvariant 3 shallower crater at e8", fontsize=10.5)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

ax = axes[1]
strata = [("DTU=0", lambda s: s==0), ("0–0.35", lambda s:(s>0)&(s<=0.35)),
          ("0.35–0.65", lambda s:(s>0.35)&(s<=0.65)), (">0.65", lambda s:s>0.65)]
x = np.arange(len(strata))
width = 0.2
slot = 0
for cond in ("full","atac"):
    for ep in (7,8):
        v3, v2 = load("rdtu_eval",cond,ep), load("macro_eval",cond,ep)
        m = v3.merge(v2,on="gene_id",suffixes=("_3","_2")).merge(dtu,on="gene_id").dropna(subset=["score_train"])
        m["d"] = m.nll_2 - m.nll_3
        vals = [m[fn(m.score_train)].d.mean() for _, fn in strata]
        color = "#1b6ca8" if cond=="full" else "#d1495b"
        ax.bar(x+(slot-1.5)*width, vals, width, color=color, alpha=1.0 if ep==7 else 0.55,
               label=f"{cond} e{ep}")
        slot += 1
ax.axhline(0, color="black", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels([n for n, _ in strata])
ax.set_xlabel("train-only DTU stratum")
ax.set_ylabel("variant2 − variant3 NLL  (pos = new loss better)")
ax.set_title("Direct effect of the new loss: gains grow with DTU,\nboth arms, e7/e8", fontsize=10.5)
ax.legend(fontsize=8.5)
ax.grid(alpha=0.25, axis="y")

fig.suptitle("R005 verdict — variant 3 vs variant 2 (seed 1103)", fontsize=12)
fig.tight_layout()
fig.savefig("outputs/analysis/rdtu_verdict.png", dpi=150)
print("outputs/analysis/rdtu_verdict.png")
