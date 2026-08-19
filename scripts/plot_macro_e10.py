"""First-10-epoch report for the gene_macro paired retraining."""
import csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR = {"full": "#1b6ca8", "atac": "#d1495b"}
def read(run):
    with open(f"runs/fabric_v2_{run}_seed1103/history.tsv") as h:
        rows = list(csv.DictReader(h, delimiter="\t"))
    return {int(r["epoch"]): (float(r["validation_compatible_path_nll_gene_macro"]),
                              float(r["validation_compatible_path_nll"]),
                              float(r["epoch_learning_rate"])) for r in rows}
new = {c: read(f"{c}_macro") for c in ("full","atac")}
old = {}
for c in ("full","atac"):
    with open(f"runs/fabric_v2_{c}_seed1103/history.tsv") as h:
        old[c] = {int(r["epoch"]): float(r["validation_compatible_path_nll"])
                  for r in csv.DictReader(h, delimiter="\t") if int(r["epoch"]) <= 10}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

ax = axes[0]
for c, d in new.items():
    es = sorted(d); ax.plot(es, [d[e][0] for e in es], "o-", color=COLOR[c], label=c, lw=2, ms=5)
for c, d in new.items():
    drops = [e for p,e in zip(sorted(d), sorted(d)[1:]) if d[e][2] < d[p][2]]
    for e in drops:
        ax.axvline(e, color=COLOR[c], ls="--", lw=1.1, alpha=0.6)
        ax.annotate(f"{c}: LR/2", (e, 0.985), xycoords=("data","axes fraction"),
                    fontsize=8, color=COLOR[c], ha="left", textcoords="offset points", xytext=(3,-8))
ax.set_title("NEW objective: macro NLL (selection metric)", fontsize=11)
ax.set_xlabel("epoch"); ax.set_ylabel("gene-macro validation NLL")
ax.legend(); ax.grid(alpha=0.25)

ax = axes[1]
shared = sorted(set(new["full"]) & set(new["atac"]))
gap_new = [new["atac"][e][0] - new["full"][e][0] for e in shared]
gap_old = [old["atac"][e] - old["full"][e] for e in sorted(set(old["full"]) & set(old["atac"])) ]
ax.bar([e-0.2 for e in shared], gap_new, 0.4, color="#4a7c59", label="NEW runs, macro gap")
ax.bar([e+0.2 for e in shared], gap_old[:len(shared)], 0.4, color="#b8b8b8", label="OLD runs, micro gap")
ax.axhline(0, color="black", lw=0.9)
ax.set_title("full-vs-atac gap: new(macro) vs old(micro)", fontsize=11)
ax.set_xlabel("epoch"); ax.set_ylabel("atac − full  (pos = full better)")
ax.legend(fontsize=8.5); ax.grid(alpha=0.25, axis="y")

ax = axes[2]
for c, d in new.items():
    es = sorted(d); ax.plot(es, [d[e][1] for e in es], "o-", color=COLOR[c], label=f"{c} (new run)", lw=2, ms=5)
for c, d in old.items():
    es = sorted(d); ax.plot(es, [d[e] for e in es], ":", color=COLOR[c], label=f"{c} (old run)", lw=1.8, alpha=0.7)
ax.set_title("micro NLL: new objective (solid) vs old (dotted)", fontsize=11)
ax.set_xlabel("epoch"); ax.set_ylabel("molecule-weighted validation NLL")
ax.legend(fontsize=8); ax.grid(alpha=0.25)

fig.suptitle("gene_macro paired retraining — first 10 epochs", fontsize=12.5)
fig.tight_layout()
fig.savefig("outputs/analysis/macro_e10.png", dpi=150)
print("outputs/analysis/macro_e10.png")
