# FABRIC cell split and ONT-first training-gene selection

## Status

| Field | Decision |
|---|---|
| Cell universe | 217,933 ONT cells from Emb01-Emb09 |
| Split | Within every embryo, deterministic cell-level 80/10/10 |
| Primary structural training catalog | 17,706 canonical nuclear genes |
| Selection evidence | Train-only raw ONT transcript counts |
| Compatible-read rebuild | `NOT_PERFORMED` |
| Formal training | `NOT_AUTHORIZED_OR_STARTED` |

This document freezes the cell split and ONT-first structural gene catalog. It
does not claim that all selected genes already contribute likelihood: positive
train-only informative compatible-read mass must still be established after the
compatibility artifact is rebuilt.

## 1. Scientific object and catalog boundary

FABRIC predicts relative probabilities over the resolved ONT-matrix isoforms
for a `(cell, gene)` under the observed long-read library. Gene admission and
the model path axis use the same matrix transcript universe:

- **gene admission starts from the ONT transcript matrix** and uses only train
  cells;
- **the model path set is exactly the matrix-matched structural path set** for
  every admitted gene;
- GTF transcripts without an ONT matrix row do not enter the model softmax,
  compatible sets, loss, PSI, evaluation, or attribution;
- DTU labels, validation cells, and test cells do not affect admission.

Primary admission requires at least two distinct matrix-matched structural
paths. Both paths must be resolved ONT matrix rows and must each have positive
raw observation in train cells; a GTF-only transcript cannot satisfy the rule.
Exact duplicate exon chains would be collapsed before counting; the current
matrix universe contains no such duplicate aliases.

## 2. Cell split

### 2.1 Rule

Use all 217,933 ONT cells. The canonical cell identity is
`RNA__<raw_ONT_barcode>`. With seed `20260725`, independently within every
`rna_embryo_id`:

1. hash canonical JSON
   `{"split_seed":20260725,"rna_embryo_id":...,"cell_id":...}` with SHA-256;
2. sort by `(stable_key_sha256, cell_id)`;
3. set `n_val=floor(0.1*n)`, `n_test=floor(0.1*n)`, and
   `n_train=n-n_val-n_test`;
4. assign the ordered cells to train, validation, and test in that order.

This reproduces the frozen ADR-0018 algorithm. Its full-cohort proposed
manifest identity is
`4432be7e6f8f043850f98b1d579b4f4a68d2c7236c6ae0529695b00a23e24c64`.

### 2.2 Counts

| Embryo | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Emb01 | 9,999 | 8,001 | 999 | 999 |
| Emb02 | 14,134 | 11,308 | 1,413 | 1,413 |
| Emb03 | 9,188 | 7,352 | 918 | 918 |
| Emb04 | 23,312 | 18,650 | 2,331 | 2,331 |
| Emb05 | 32,061 | 25,649 | 3,206 | 3,206 |
| Emb06 | 29,118 | 23,296 | 2,911 | 2,911 |
| Emb07 | 44,724 | 35,780 | 4,472 | 4,472 |
| Emb08 | 23,128 | 18,504 | 2,312 | 2,312 |
| Emb09 | 32,269 | 25,817 | 3,226 | 3,226 |
| **Total** | **217,933** | **174,357** | **21,788** | **21,788** |

The old 167,235-cell split cannot be extended by appending cells: 309 shared
cells change assignment when the within-embryo quantile boundaries are
recomputed on the full cohort. The full 217,933-cell split must therefore be
frozen as a new identity.

This design estimates transductive supervised cell-holdout performance within
the nine observed embryos. Because cells from every embryo occur in every
split, it does not test generalization to an unseen embryo or donor.

## 3. Complete ONT transcript identity

The ONT matrix has 101,067 transcript rows and 217,933 cell columns. The old
`transcript_feature_index.parquet` resolved 36,023 rows and marked 65,044 as
`custom_name_mapping_pending`. "Pending" did not mean absent from the GTF. The
complete deterministic crosswalk is:

| Mapping rule | Rows |
|---|---:|
| Existing GENCODE v32 transcript-name map | 36,023 |
| `nrg_nr_######` or `*_nr-######` to `novel_transcript_######` | 63,808 |
| Unique punctuation-normalized GENCODE transcript name | 1,230 |
| Embedded `*_nr-ENST...` stable ID | 6 |
| **Total** | **101,067** |

All 101,067 matrix rows map one-to-one onto the 101,067 transcripts in
`transcripts_filtered.gtf`. Every mapped transcript is also present in the full
GTF with identical gene, chromosome, strand, and ordered exon chain. Therefore
none of the 65,044 formerly pending rows is excluded.

## 4. Train-only ONT evidence

The matrix contains raw positive integer molecule/count observations, not
normalized expression.

| Quantity | Full matrix | Train cells only |
|---|---:|---:|
| Cells | 217,933 | 174,357 |
| Transcript rows | 101,067 | 101,067 observed |
| Nonzero entries | 139,980,703 | 112,002,723 |
| Raw count sum | 318,469,006 | 254,731,664 |

Every transcript row has at least 3 raw counts in at least 3 positive train
cells. Validation and test counts were not read by the admission rule.

The matrix-matched GTF contains 28,002 genes. Of these, 27,957 occur on one
canonical nuclear chromosome and one strand on `chr1-22`, `chrX`, or `chrY`; 30 are
nuclear alt-contig genes and 15 are mitochondrial genes.

## 5. Selected gene tiers

### 5.1 Primary catalog

Define

\[
G_{ONT}^{primary}
=\{g:\text{canonical nuclear and at least 2 distinct matrix paths have}
\; count_{train}>0\}.
\]

The resulting primary structural training catalog contains **17,706 genes**
and **90,672 ONT matrix structural paths**. These 90,672 paths are the complete
model isoform axis used by compatible-path likelihood, probability
normalization, PSI, evaluation, and attribution. No full-GTF-only path is added.

The catalog is intentionally not filtered by gene span, ATAC availability,
DTU score, cell-type label confidence, or arbitrary 5/10-count thresholds.

### 5.2 Explicit secondary and audit tiers

| Tier | Genes | Use |
|---|---:|---|
| Canonical genes with at least 2 train-observed ONT paths | 17,706 | primary catalog |
| Exactly 2 matrix paths | 4,348 | included binary-choice genes |
| Alt-contig genes with at least 3 paths | 13 | conditional after locus/reference QC |
| Mitochondrial genes with at least 3 paths | 2 | audit only |

Exactly-two-path genes are included because both isoforms are matrix rows with
positive train observation. Their likelihood is informative only when a
positive-weight compatible set is a singleton; a full two-path compatible set
remains audit-only under the same \(\mathcal K^{inf}\) rule.

### 5.3 Actual likelihood membership

After compatible-read reconstruction, the gradient-bearing training set is

\[
G_{fit}^{likelihood}
=\left\{g\in G_{ONT}^{primary}:
\sum_{i\in C_{train}} w^{inf}_{i,g}>0\right\},
\]

where `w_inf` contains only positive-weight compatible sets that are nonempty
proper subsets of the gene's matrix structural-path set. The current historical EC artifact
covers only 7,198 genes and cannot be used to assert the size of this new set.
Validation/test evidence cannot admit a gene or upgrade its reporting tier.

## 6. Frozen DTU gene prior

`data/DTU_result_sorted.xlsx` contains the same 28,002 genes as the
matrix-matched GTF, and its `number_of_transcripts` agrees exactly for every
gene. This workbook is the frozen DTU gene prior; FABRIC does not recompute a
train-only DTU score and does not place an additional provenance gate in front
of this prior.

There are 2,844 `top_DTU_gene=yes` genes. The primary catalog contains all 2,841
canonical high-DTU genes; the remaining 3 are alt-contig genes. DTU metadata
was joined only after selection and did not control gene admission or support
thresholds. Downstream sensitivity runs may use the continuous `DTU_score` as
the frozen prior defined in the architecture contract. Its historical score
implementation is available in `data/DTU_score.R`: it first restricts PSI,
expression, and transcript-to-gene metadata to their common transcript axis,
requires at least two transcripts and at least two expressed cell types, and
scores dominant-transcript switching using transcript-wise JS divergence. The
workbook flag is exactly equivalent to `DTU_score >= 0.7` in the delivered
28,002-gene table, but that threshold assignment is not present in the R file.

All 4,361 two-transcript genes in the workbook are labelled non-high-DTU under
the delivered 0.7 cutoff. This is not used as a reason to exclude them from
FABRIC: the 4,348 canonical genes whose two transcripts are both train-observed
ONT matrix rows are admitted to the primary structural catalog.

## 7. Reproducible outputs

The selection implementation is `src/fabric/ont_gene_selection.py`. The local
derived package is `data/processed/fabric_ont_gene_selection_v3/` and contains:

- `selected_ont_training_gene_catalog.tsv`: 17,706 primary genes;
- `gene_selection_audit.parquet`: all 28,002 genes and support/tier fields;
- `transcript_crosswalk_audit.parquet`: all 101,067 ONT rows and mapping rules;
- `split_rows.parquet` and `matrix_cell_index.parquet`: frozen cell assignment;
- `split_summary.tsv` and `selection_summary.json`: counts, identities, and
  scientific status.

Primary evidence sources are:

- `/home2/xyf/project/PRISM/data/matrix/tx_matrix_ONT/tx_matrix_ONT.mtx`;
- `/home2/xyf/project/PRISM/data/matrix/tx_matrix_ONT/transcripts.tsv`;
- `/home2/xyf/project/PRISM/data/matrix/tx_matrix_ONT/barcodes.tsv`;
- `/home2/xyf/project/PRISM/data/gtf/transcripts_filtered.gtf`;
- `data/DTU_score.R`;
- `data/DTU_result_sorted.xlsx`.

This selection does not authorize or start formal training.
