# FABRIC — 仓库布局与产物读写约定

FABRIC（Factor-Aware Branch Regulation of Isoform Choice）是一个基因级图模型，
从单细胞长读长（ONT）+ Illumina RNA + ATAC 数据建模 isoform 选择。
权威设计契约见 [docs/FABRIC_ARCHITECTURE_V2.md](docs/FABRIC_ARCHITECTURE_V2.md)
（V1 仅作历史参考）。研究过程记录（提案、实验计划、tracker、外部评审）
在 [refine-logs/](refine-logs/)。

本文件定义**各类东西放在哪、agent 可以在哪读写**。
任何在本仓库工作的 agent（Claude Code、Codex 等）都必须遵守。

## 目录地图

| 路径 | Git | 是什么 | Agent 规则 |
|---|---|---|---|
| `src/fabric/` | tracked | 主 package。入口：`train.py`、`evaluate.py`。`*_real*` 模块是真实数据路径；`dataset.py`/`graph.py` 等 toy 与 real 共用。 | 正常改代码。单文件很大（3–4k 行），优先做外科手术式小改动。 |
| `tests/` | tracked | Pytest 测试套件。`tests/fixtures/real/` 存放已提交的小型 parquet fixture。 | 新测试按 `test_<module>.py` 的命名模式与模块对应。 |
| `configs/` | tracked | YAML 运行配置。每个文件是**一次已授权运行设置的记录**（带 `training_authorized` / `final_test_authorized` 标志）。 | 新运行 ⇒ 新文件。已完成运行用过的 config 永不修改。 |
| `docs/` | tracked | 架构契约文档。 | V2 文档是权威；只有真实设计变更才更新它。 |
| `scripts/` | tracked | 事后分析、绘图与运维脚本（top-1 评估、DTU 重算、checkpoint 快照守护进程）。 | **必须从 repo 根目录运行**——相对路径默认 CWD 为 repo 根。新分析代码写在这里（并提交），绝不放 `tmp/`。产物写入 `outputs/analysis/`。 |
| `refine-logs/` | tracked | FINAL_PROPOSAL.md、EXPERIMENT_PLAN.md、EXPERIMENT_TRACKER.md、评审轮次记录。 | 实验推进时追加/更新 tracker 行。 |
| `sources/` | tracked | 文献调研笔记（markdown）。 | 新调研笔记放这里。 |
| `data/` | 混合 | 见下方细分。 | |
| `runs/` | **ignored** | 训练输出，每次运行一个目录：`fabric_v2_<variant>_seed<seed>/`，旁边是同名 `<name>.log`。`runs/checkpoint_snapshots/<run>/epoch_N.pt` 保存逐 epoch 权重（train.py 会原地覆盖 `latest.pt`）。 | **只增不删。**永不删除、重命名或覆盖已有的 run 目录或快照。只有 `train.py`（或快照守护进程）向这里写入。 |
| `outputs/` | **ignored** | `validation/` = 就绪/授权验证日志；`analysis/` = `scripts/` 产出的衍生分析产物（per-gene TSV、top-1 JSONL、图 PNG）。 | 可再生但需消耗 GPU——不要随手删。只有分析脚本向这里写入。 |
| `tmp/` | **ignored** | 真正的草稿区。随时可删，删了无损失。 | 任何丢了会心疼的东西都**不**属于这里。 |
| `paper/` | **ignored** | 外部参考 PDF / 合作者手稿（输入材料，不是产物）。 | 只读参考材料。 |

### `data/` 细分

| 路径 | Git | 是什么 |
|---|---|---|
| `data/external_inputs.yaml` | tracked | 所有外部输入路径的**唯一事实源**（ONT 矩阵、Illumina RNA、ATAC peaks、GLUE embedding、参考 FASTA/GTF——指向 PRISM / Multi_Omic 项目的绝对路径）。 |
| `data/DTU_score.R`、`data/DTU_result_sorted.xlsx` | tracked | 原始 DTU 参考分数及其 R 源码（`scripts/recompute_dtu.py` 的保真校验目标）。 |
| `data/processed/` | ignored | 带版本号的衍生产物（如 `fabric_ont_gene_selection_v3`、`fabric_v2_compatible_ec_v1`），各自带 manifest JSON。**建成即不可变**——有变更就升新版本号（`…_v2`），绝不原地修改。 |
| `data/data_cpu/` | ignored | 约 13 GB 的外部矩阵本地镜像，供 CPU 侧工作使用。自带 README。对 agent 只读。 |
| `data/gate_baselines/`、`data/splits/` | （空） | 代码创建的输出落点。 |

## 产物读写规则

1. **什么东西写到哪**
   - 新分析/绘图脚本 → `scripts/`（提交）。其数据产物 → `outputs/analysis/`。
   - 新训练运行 → `configs/` 里新建 YAML，输出经由 `src/fabric/train.py` 落到 `runs/<run_name>/`。
   - 真正的一次性草稿 → `tmp/`（或会话 scratchpad），别的地方都不行。
   - 设计决策 → `docs/`；实验状态 → `refine-logs/EXPERIMENT_TRACKER.md`。

2. **Git 卫生**
   - `runs/`、`outputs/`、`tmp/`、`paper/`、`data/processed/`、`data/data_cpu/` 是**有意** gitignore 的。永不 `git add -f` 它们；永不盲目 `git add .`——先看 `git status`。
   - 脚本和配置要与其产出的结果一起提交，保证每张图/每张表都能从已追踪的生成器再生。

3. **外部数据只读。**原始矩阵在本仓库之外（绝对路径见 `data/external_inputs.yaml`）。永不向 PRISM 或 Multi_Omic 项目树写入。脚本需要原始输入时，从 `external_inputs.yaml` 取路径，不要另行硬编码。

4. **测试集纪律。**held-out test 的暴露由架构文档管辖。config 带显式授权标志；test-compatible 行是刻意不物化的。除非存在 `final_test_authorized: true` 的 config 且用户明确要求，**永不**计算、缓存或打印测试集预测/指标。

5. **运行命名。**训练运行遵循 `fabric_v2_<variant>_seed<seed>`（如 `fabric_v2_atac_macro_seed1103`）。保持该模式；分析脚本会解析它。

## 如何运行

- 环境：`pyproject.toml`（`pip install -e .` 安装）；在 repo 根目录跑 `pytest`（GPU 冒烟测试在 `tests/test_gpu_smoke.py`；多数契约测试仅需 CPU）。
- 训练/评估：用 `configs/` 里的配置直接执行 `src/fabric/train.py` / `evaluate.py`（脚本自行把 `src/` 插入 `sys.path`）。
- 分析：从 repo 根目录运行 `scripts/*.py`；它们读取 `runs/` + `outputs/analysis/`，并写回 `outputs/analysis/`。
