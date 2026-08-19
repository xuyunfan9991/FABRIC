# FABRIC — Repository Layout & Artifact Contract

FABRIC (Factor-Aware Branch Regulation of Isoform Choice) is a gene-level graph
model of isoform choice from single-cell long-read (ONT) + Illumina RNA + ATAC
data. The authoritative design contract is
[docs/FABRIC_ARCHITECTURE_V2.md](docs/FABRIC_ARCHITECTURE_V2.md) (V1 is
historical). Research history (proposal, experiment plan, tracker, external
reviews) lives in [refine-logs/](refine-logs/).

This file defines **where things live and where agents may read and write**.
Any agent (Claude Code, Codex, …) working in this repo must follow it.

## Directory map

| Path | Git | What it is | Agent rules |
|---|---|---|---|
| `src/fabric/` | tracked | The package. Entrypoints: `train.py`, `evaluate.py`. `*_real*` modules are the real-data path; `dataset.py`/`graph.py` etc. serve both toy and real. | Normal code changes. Files are large (3–4k lines); prefer surgical edits. |
| `tests/` | tracked | Pytest suite. `tests/fixtures/real/` holds small committed parquet fixtures. | Add tests next to the module naming pattern `test_<module>.py`. |
| `configs/` | tracked | YAML run configs. Each file is a **record of an authorized run setup** (carries `training_authorized` / `final_test_authorized` flags). | New run ⇒ new file. Never mutate a config that a finished run used. |
| `docs/` | tracked | Architecture contracts. | V2 doc is authoritative; update it only for real design changes. |
| `scripts/` | tracked | Post-hoc analysis, plotting, and ops scripts (top-1 eval, DTU recompute, checkpoint snapshot daemon). | **Run from repo root** — relative paths assume CWD = repo root. New analysis code goes here (and gets committed), never in `tmp/`. Outputs go to `outputs/analysis/`. |
| `refine-logs/` | tracked | FINAL_PROPOSAL.md, EXPERIMENT_PLAN.md, EXPERIMENT_TRACKER.md, review rounds. | Append/update tracker rows as experiments progress. |
| `sources/` | tracked | Literature survey notes (markdown). | Add new survey notes here. |
| `data/` | mixed | See breakdown below. | |
| `runs/` | **ignored** | Training outputs, one dir per run: `fabric_v2_<variant>_seed<seed>/` plus a sibling `<name>.log`. `runs/checkpoint_snapshots/<run>/epoch_N.pt` preserves per-epoch weights (train.py overwrites `latest.pt` in place). | **Append-only.** Never delete, rename, or overwrite an existing run dir or snapshot. Only `train.py` (or the snapshot daemon) writes here. |
| `outputs/` | **ignored** | `validation/` = readiness/authorization logs; `analysis/` = derived analysis products (per-gene TSVs, top-1 JSONLs, figures PNGs) written by `scripts/`. | Regenerable but GPU-expensive — do not casually delete. Analysis scripts write here, nothing else does. |
| `tmp/` | **ignored** | True scratch. Deletable at any time without loss. | Anything you'd mind losing does **not** belong here. |
| `paper/` | **ignored** | External reference PDFs / collaborator manuscripts (inputs, not products). | Read-only reference material. |

### `data/` breakdown

| Path | Git | What it is |
|---|---|---|
| `data/external_inputs.yaml` | tracked | **Single source of truth** for all external input paths (ONT matrix, Illumina RNA, ATAC peaks, GLUE embedding, reference FASTA/GTF — absolute paths into the PRISM / Multi_Omic projects). |
| `data/DTU_score.R`, `data/DTU_result_sorted.xlsx` | tracked | Original DTU reference score + its R source (fidelity target for `scripts/recompute_dtu.py`). |
| `data/processed/` | ignored | Versioned derived artifacts (e.g. `fabric_ont_gene_selection_v3`, `fabric_v2_compatible_ec_v1`), each with a manifest JSON. **Immutable once built** — a change means a new version id (`…_v2`), never editing in place. |
| `data/data_cpu/` | ignored | ~13 GB local mirror of external matrices for CPU-side work. Has its own README. Read-only for agents. |
| `data/gate_baselines/`, `data/splits/` | (empty) | Output targets created by code. |

## Rules for reading and writing artifacts

1. **Where to write what**
   - New analysis / plotting script → `scripts/` (committed). Its data products → `outputs/analysis/`.
   - New training run → new YAML in `configs/`, outputs land in `runs/<run_name>/` via `src/fabric/train.py`.
   - Genuine throwaway scratch → `tmp/` (or the session scratchpad), nowhere else.
   - Design decisions → `docs/`; experiment status → `refine-logs/EXPERIMENT_TRACKER.md`.

2. **Git hygiene**
   - `runs/`, `outputs/`, `tmp/`, `paper/`, `data/processed/`, `data/data_cpu/` are gitignored **on purpose**. Never `git add -f` them; never blind `git add .` — check `git status` first.
   - Commit scripts and configs alongside the results they produced, so every figure/table is regenerable from a tracked generator.

3. **External data is read-only.** Raw matrices live outside this repo (absolute paths in `data/external_inputs.yaml`). Never write into the PRISM or Multi_Omic project trees. When a script needs a raw input, take the path from `external_inputs.yaml` rather than hardcoding a new one.

4. **Test-set discipline.** The architecture doc governs held-out test exposure. Configs carry explicit authorization flags; test-compatible rows are deliberately not materialized. **Never** compute, cache, or print test-set predictions/metrics unless a config with `final_test_authorized: true` exists and the user explicitly asks.

5. **Run naming.** Training runs follow `fabric_v2_<variant>_seed<seed>` (e.g. `fabric_v2_atac_macro_seed1103`). Keep the pattern; the analysis scripts parse it.

## Running things

- Environment: `pyproject.toml` (install with `pip install -e .`); tests via `pytest` from repo root (GPU smoke tests are in `tests/test_gpu_smoke.py`; most contract tests are CPU-only).
- Training/eval: `python -m` or direct file execution of `src/fabric/train.py` / `evaluate.py` with a config from `configs/` (scripts insert `src/` into `sys.path` themselves).
- Analysis: run `scripts/*.py` from repo root; they read `runs/` + `outputs/analysis/` and write back into `outputs/analysis/`.
