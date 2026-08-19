# Experiment Plan

**Problem**: Test whether context-module gains are diluted by genes with stable
transcript usage, without redefining the task around an outcome-selected subset.
**Method Thesis**: Keep the full gene universe for CIS learning and general
evaluation, but train the dynamic State/DNA/RNA and future choice-pair heads with
train-only, modality-specific DTU-aware gene weighting and evaluate locked DTU
strata separately.
**Date**: 2026-08-11

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: Regulatory context helps mainly where isoform usage changes across states | Explains a weak all-gene average without hiding failures | Full-minus-CIS delta NLL becomes more favorable with train-defined DTU, and exceeds shuffled-context gain | B1 |
| C2: DTU-aware training recovers context signal without hallucinating it in stable genes | Supports the proposed training estimand | Better high-DTU held-out NLL and event attribution, with near-null context gain on stable genes and reported all-gene performance | B2-B4 |
| Anti-claim: gain is only expression reweighting or outcome cherry-picking | Separates mechanism from sampling artifact | Macro-without-DTU control, locked train-only strata, all-gene and stable-gene endpoints | B2-B4 |

## Paper Storyline

- Main paper must prove: context value is concentrated in biologically responsive,
  regulator-addressable genes; DTU-aware weighting improves that conditional task.
- Appendix can support: high-DTU-only upper bound, threshold sensitivity, and
  alternative shrinkage estimators.
- Intentionally cut: a sole high-DTU-only benchmark or thresholds recomputed on
  validation/test outcomes.

## Experiment Blocks

### B1: No-retraining dilution diagnosis

- Claim tested: C1.
- Dataset / split: current locked cell/donor split; estimate DTU from training
  outcomes only, then apply the locked gene strata to validation/test.
- The current cell holdout is transductive within embryos. Add leave-one-embryo-out
  as the mechanism-generalization endpoint; cells are not biological replicates.
- Compared systems: CIS, observed-context Full, and context-shuffled Full.
- Metrics: per-gene `delta_NLL = NLL_Full - NLL_CIS`; macro and molecule-weighted
  summaries by DTU decile; observed-minus-shuffled delta.
- Success criterion: a reproducible monotone or thresholded improvement in the
  high-DTU strata, with confidence intervals based on genes and donors.
- Failure interpretation: if the trend is absent, low-DTU dilution is not the main
  explanation; inspect routing, feature noise, and optimization instead.
- Priority: MUST-RUN.

### B2: Separate expression-mass dilution from DTU dilution

- Claim tested: C2 and anti-claim.
- Freeze the identical full-gene CIS parent for every comparison.
- Dynamic-head objectives:
  1. current global molecule-weighted objective;
  2. per-gene macro objective without DTU weighting;
  3. per-gene macro objective with bounded, continuous DTU weighting;
  4. high-DTU-only training as an appendix upper bound.
- Use the same seeds, budget, eligible choices, and checkpoints.
- Success criterion: variant 3 improves the locked high-DTU endpoint beyond variant
  2, not merely beyond the molecule-weighted baseline.
- Priority: MUST-RUN.

### B3: Modality-matched DTU

- Claim tested: regulatory features help the choices they can biologically address.
- RBP score: internal splice/exon-choice DTU plus dynamic RBP gate and routed RNA
  motif opportunity.
- ATAC score: TSS DTU, PAS DTU reported separately, plus dynamic local accessibility;
  promoter accessibility-only and TF-DNA motif events remain distinct.
- Future pair head: require both context-dependent usage and pair residual-rank
  identifiability; generic high DTU is insufficient.
- Success criterion: modality-matched strata show stronger observed-over-shuffled
  gains than generic path-DTU strata.
- Priority: MUST-RUN.

### B4: Stable-gene null and attribution validation

- Claim tested: C2.
- Define stable genes by a small effect-size equivalence bound with adequate power,
  not by a nonsignificant DTU p-value.
- Metrics: context delta NLL, magnitude of dynamic logits, calibration, event-mask
  probability delta, and perturbation/case-study concordance when available.
- Success criterion: context corrections stay close to zero on stable genes while
  high-DTU genes show reproducible, directionally coherent effects.
- Priority: MUST-RUN.

## Train-only DTU Definition

For gene `g`, estimate the state-specific legal-path distribution `pi[g,s]` from
training long-read compatible sets using shrinkage. Use the generalized Jensen-
Shannon effect size

`D_path[g] = sum_s w[s] KL(pi[g,s] || pi[g])`,

and calculate analogous marginal scores for TSS, PAS, and internal choices.
Require minimum informative molecule mass and replication across donors. Prefer a
continuous rank-normalized score for training weights; thresholds are for reporting.

For a dynamic head, use a mixed two-level objective

`L_context = (1-lambda) * L_micro_all + lambda * L_macro_DTU`,

where

`L_macro_DTU = sum_g w[g] * (sum_k-in-g n[k] * loss[k] / sum_k-in-g n[k]) / sum_g w[g]`,

where `w[g]` is bounded away from zero so stable genes remain negative controls.
Choose `lambda` inside training only and use the identical objective for every
dynamic ablation variant.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Relative Cost | Risk |
|---|---|---|---|---|---|
| M0 | Validate DTU estimator and locked strata | donor bootstrap, low/high-support sanity cases | scores stable across donors and train resamples | Low | sparse long-read support |
| M1 | Test dilution without retraining | B1 | high-DTU trend and observed-over-shuffled gain | Low | apparent trend driven by expression |
| M2 | Isolate loss weighting | B2 variants 1-3, paired seeds | DTU weighting beats macro-only control | One dynamic-head training budget per variant | calibration shift |
| M3 | Match modality and choice type | B3 | RBP/ATAC gains align with addressable events | Moderate | small eligible strata |
| M4 | Null and interpretation checks | B4 plus high-only appendix | stable genes remain near-null | Moderate | winner's curse in DTU score |

## Evaluation Endpoints

- Full gene universe: general prediction and calibration.
- Locked train-defined high-DTU universe: primary context-mechanism endpoint.
- Stable, adequately powered universe: false-context-effect endpoint.
- Modality-opportunity universe: RBP- and ATAC-specific mechanism endpoint.
- Pair-identifiable high-DTU universe: future choice-coupling endpoint.
- Report both micro molecule-weighted NLL and macro per-gene NLL.
- Report the existing within-embryo cell holdout separately from leave-one-embryo-
  out; the former does not establish new-embryo generalization.

## Risks and Mitigations

- Target leakage: split cells/donors first; fit DTU and thresholds only on training
  outcomes; never update the list after seeing validation/test DTU.
- Expression confounding: include the macro-without-DTU objective.
- Low-count winner's curse: shrink estimates, require support, and bootstrap donors.
- Cell-composition confounding: condition DTU on the declared cell state and verify
  within-donor replication.
- Changed estimand: retain all-gene and stable-gene evaluation; label the high-DTU
  result explicitly as conditional.

## Final Checklist

- [x] Main paper claim and anti-claim are explicit
- [x] No-retraining falsification precedes new training
- [x] High-DTU labels are train-only and locked
- [x] Stable genes remain negative controls
- [x] Micro and macro objectives are separated
- [x] Modality-specific and pair-identifiable strata are separated
