# runs/ 目录清单

`runs/` 本身 gitignore 且只增不删；本文件是每个 run 目录的用途与处置的权威记录。
每个 `fabric_v2_*` run 目录内的 `config.yaml` + `training_run_manifest.json` 是其自描述；
本清单负责"为什么跑、结局如何"。新增 run 时追加一行。

| 目录 | 日期 | 角色 | Epochs | 状态与处置 |
|---|---|---|---|---|
| `toy_cpu_initial` | 08-10 | V1 toy 首次 CPU 冒烟 | 400 | 完成，历史参考 |
| `toy_{cpu,gpu0,gpu1}_v1_{validation,contract_final,final}`（9 个） | 08-11 | V1 契约验证套件：3 平台 × 3 阶段 | 各 400 | 完成，历史参考 |
| `fabric_v2_full_seed1103` / `fabric_v2_atac_seed1103` | 08-18 | **R003 / B2 变体 1**：molecule-weighted 基线配对 | 30 | 完成。产出指标脆弱性发现（官方 NLL 被 RPS25 主导、macro 口径结论反转、高 DTU 基因 e19→e30 退化）；per-gene 证据在 `outputs/analysis/per_gene_*_e{19,30}.tsv` |
| `fabric_v2_full_macro_seed1103` / `fabric_v2_atac_macro_seed1103` | 08-18→19 | **R004 / B2 变体 2**：gene_macro 等权配对 | 10 | e10 人为停止；经决定（08-20）**不续跑**，e10 窗口即变体 2 基线。产出 dose-response 复测：高 DTU 增益在 e7/e10 复现（+0.005）、e8 大幅反转（−0.023）、epoch 波动 σ≈0.005 与效应同阶。逐 epoch 权重在 `runs/checkpoint_snapshots/{full,atac}_macro/` |
| `fabric_v2_{full,atac}_reliability_dtu_macro_seed1103` | 08-20 | 变体 3 首次启动的**流产遗迹**：被 clean-tree 门禁拦下（src/fabric 未提交） | 0 | 无训练数据；按只增不删保留，勿复用目录名 |
| `fabric_v2_{full,atac}_reliability_dtu_macro_lrhold10_seed1103` | 08-20 | **R005 / B2 变体 3**：reliability_dtu_macro 配对（τ=100、α=1、LR 5e-5 前 10 epoch 恒定） | 0 | 用户于 18:09 在 e1 完成前停止（与 GLUE 同机竞争致 epoch 过慢）；无 checkpoint 无指标，目录仅剩 config/manifest 存根。重启时**换新目录名**（勿复用），此对按只增不删保留 |
| `fabric_v2_{full,atac}_reliability_dtu_macro_lrhold10_omp16_seed1103` | 08-20→21 | **R005 / B2 变体 3**：`OMP/MKL=16` + taskset 绑核（full 0–15/GPU0、atac 64–79/GPU1），~1h20–50/epoch | full e9 / atac e8 | 08-21 15:0x 用户要求停止（未到 e10）。逐 epoch 权重在各自 `epoch_checkpoints/`；同 commit 可从 `latest.pt` 续跑。无权 gene_macro 口径逐 epoch 略优于变体 2（~0.004–0.007）；进度图 `outputs/analysis/rdtu_progress.png` |
| `fabric_v2_{full,atac,rbp}_rdtu_a0_shared6g2x_seed1103` | 08-23 | **R012 / 三臂共卡 2× 容量家族**：reliability_dtu_macro **α=0**、hidden 128/dynamic 64/heads 8/path 128、共卡 LR 协议（plateau patience=2 无 hold）。full 独占 GPU0，atac+rbp 共 GPU1；画像 `fabric_v2_shared6g2x_resources_v1` + passing canary | full e1 / atac e0 / rbp e0 | 约 1.5 h 后与同机 GLUE 一起被外部 SIGKILL，原因未能从当前账户确定；Full 的 `latest.pt`/`best.pt`/`epoch_1.pt` 可加载，可在原目录从 `latest.pt` 恢复。ATAC/RBP 无 checkpoint，旧目录只增不删且不得复用，重启必须使用新 run identity。`scripts/launch_shared6g2x.sh` 保留首次启动命令并拒绝复用已存在目录。该家族与 R003–R005 不作数值配对；final test 未访问 |
| `checkpoint_snapshots/` | 08-18→19 | 已退役快照守护进程的产物（非 run）：`{full,atac}/` = R003 的后期窗口（full e18–e30、atac e19–e30）；`{full,atac}_macro/` = R004 的 e1–e10 | — | 只增不删保留；新 run 不再使用此机制 |
