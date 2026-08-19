# Round 1 External Review

**Reviewer verdict**: REVISE, 6.5/10  
**Review focus**: statistical validity, endpoint evidence, identifiability, attribution semantics, and leakage.

## Blocking issues identified

1. Pair admission must use within-state/cross-fitted residuals; double-centering alone does not remove coupling induced by mixed cell states.
2. Alternative TSS/PAS require explicit 5-prime completeness, cap/annotation support, poly(A)-signal and internal-priming QC.
3. Long-read length and terminal-completeness bias can confound long-range pair support; establish an observation-bias policy before making TSS-PAS claims.
4. Pair orientation must be a fixed transcript-order prior, not a learned causal direction.
5. Event masking must state whether unary and pair routes are both removed; the revised plan removes both.
6. Stable controls must be adequately powered and matched on expression, depth, length, path/choice complexity and event opportunity.
7. DTU labels and weights must be train-only; the external list cannot silently feed training/model selection.
8. Pair/event families require held-out estimation and multiple-testing control after train-time admission.

## Simplifications accepted

- Pair admission and PairScorer are one coherent pipeline.
- Accessibility caveats are expressed once as `openness != occupancy/activation`.
- All high-capacity alternatives are one deferred/upper-bound bucket.
- Coupling-versus-causality semantics are defined once in the V2 contract.

## Resolution

All eight blocking issues were incorporated into `refine-logs/FINAL_PROPOSAL.md` without adding a new black-box model. The proposal remains `REVISE` until the high-DTU list, endpoint/capture QC and held-out DTU/pair diagnostics are run.
