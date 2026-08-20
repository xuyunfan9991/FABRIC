# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | Fit train-only path/TSS/PAS/internal DTU | shrinkage estimator | train only | DTU, support, donor stability | MUST | TODO | lock scores and strata before evaluation |
| R002 | M1 | Test dilution without retraining | CIS vs Full vs shuffled-context Full | val/test | delta NLL by DTU decile, micro/macro | MUST | TODO | no model changes |
| R003 | M2 | Current objective baseline | global molecule-weighted dynamic head | train/val/test | full, high-DTU, stable NLL | MUST | TODO | identical frozen CIS parent |
| R004 | M2 | Test expression-mass dilution | per-gene macro dynamic head | train/val/test | same as R003 | MUST | TODO | no DTU weighting |
| R005 | M2 | Test DTU-specific weighting | bounded DTU-weighted macro head | train/val/test | same plus calibration | MUST | RUNNING | Full=`fabric_v2_full_reliability_dtu_macro_lrhold10_seed1103`; ATAC=`fabric_v2_atac_reliability_dtu_macro_lrhold10_seed1103`; seed 1103; LR 5e-5 fixed for epochs 1-10; validation only, final test unauthorized |
| R006 | M3 | Match RBP to internal DTU | RBP head | val/test | observed-vs-shuffled delta NLL | MUST | TODO | require routed RNA event opportunity |
| R007 | M3 | Match ATAC to terminal DTU | DNA/ATAC head | val/test | TSS and PAS endpoints separately | MUST | TODO | accessibility-only and TF-motif strata separate |
| R008 | M4 | Stable-gene null | selected best dynamic head | val/test | logit magnitude, NLL, calibration | MUST | TODO | stability by equivalence bound |
| R009 | M4 | High-DTU-only upper bound | high-only dynamic head | val/test | conditional NLL | NICE | TODO | appendix, never sole headline |
| R010 | M4 | Choice-coupling scope | pairwise CRF | val/test | pair residual NLL, event attribution | NICE | TODO | only pair-identifiable high-DTU genes |
| R011 | M4 | Test embryo generalization | selected variants | embryo-LOO | per-embryo micro/macro NLL and paired gain | MUST | TODO | all cells and molecules from one embryo stay together |
