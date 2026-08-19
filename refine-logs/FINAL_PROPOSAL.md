# FABRIC V2：系统修改方案

**日期**：2026-08-11  
**状态**：设计已收束；尚未修改运行代码；师兄 high-DTU 列表仍待实际文件与 provenance 审计。

## 1. Problem Anchor

FABRIC 必须利用 paired long-read/short-read RNA 与映射 ATAC，回答：

> 哪个 factor 的哪个 DNA 或 RNA motif event，在什么细胞状态下，使哪个局部 RNA-processing alternative 的相对 logit 改变多少，并最终影响哪些完整转录本路径。

模型必须表达 TSS、内部剪接/exon、PAS choices 的联合依赖，而不是把每个局部 choice 当作条件独立。论文主线保持：先展示 ATAC 图谱与生物学性质，再由 FABRIC 将 ATAC 与 alternative RNA processing 和完整 isoform paths 连接起来。

## 2. 一句话方法

将 FABRIC 从 legal paths 上的 unary additive model，升级为：

> **processing graph 定义合法结构，choice graph 定义可检验依赖，cell-conditioned sparse pairwise CRF 在合法完整 paths 上给出可分解概率。**

## 3. 主贡献、支持贡献和非目标

### 主贡献

一个可审计的、event-conditioned choice-graph CRF：同时输出局部 unary effect、choice-pair conditional coupling 和完整 path effect。

### 支持贡献

一个不改变 full-gene 总体的 DTU-aware 训练与评估协议，使真正 context-responsive 的基因不会被高表达稳定基因淹没。

### 明确非目标

- 不从 RNA/ATAC/sequence 猜测 TF-RBP 蛋白物理互作；
- 不做 WT1-U2AF65、SPI1-NONO 等 PPI trainable 分支；
- 不做 all-to-all choice graph、三阶 factor 或 path-specific free parameters；
- 不把 dynamic GraphGPS、path Transformer 或 world model 作为主模型；
- 不把 ATAC openness 写成 TF occupancy；
- 不把 observational coupling 写成因果传播。

## 4. 总体结构

```text
reviewed processing graph + legal path catalog
                  │
                  ├── static CIS GraphGPS / sequence features
                  │
                  └── terminal + internal choice catalog
                                   │
                                   └── sparse directed choice graph

State / promoter accessibility / TF-DNA events / RBP-RNA events
                  │
                  ├── unary choice potentials
                  └── admitted choice-pair potentials
                                   │
                                   ▼
                         complete legal-path logits
                                   │
                                   ▼
                         exact compatible-path NLL
```

对于细胞 `i` 和合法 path `p`：

\[
L_{i,p}
=
L^{CIS}_{p}
+
\sum_c \phi_{i,c,a_c(p)}
+
\sum_{(c,d)\in E_{pair}}
\psi_{i,cd,a_c(p),a_d(p)}.
\]

其中 `phi` 是单 choice 主效应，`psi` 是超过两个 unary effects 的额外 coupling。compatible-path likelihood、gene 内 softmax 和 invalid-path hard mask 保持不变。

## 5. 必须修改一：新建 V2 科学合同

新增 `docs/FABRIC_ARCHITECTURE_V2.md` 和独立 V2 config，明确：

- 预测对象包含 alternative TSS/initiation、internal processing 和 PAS/termination；
- true terminal choice、choice-pair、pair gauge、pair admission、event routing 的定义；
- full path 的 unary + pair energy 方程；
- train-only DTU、mixed estimand、评估 scopes 和 split；
- association、mechanism consistency 与 causal validation 的语言边界；
- V2 的固定消融层级、初始化和 admission gates。

V1 文档与历史 run 保留为只读记录；V2 成为 active contract 后不维护 V1/V2 双 runtime、adapter 或 checkpoint migration。

## 6. 必须修改二：建立真正的 TSS 和 PAS choices

当前 bubble 的 `scope=tss/pas` 不能代表多个 terminal sites 之间的选择。V2 必须从 reviewed path endpoints 构建：

```text
choice_kind = terminal_tss | internal | terminal_pas
TSS alternatives = reviewed TSS/promoter clusters
PAS alternatives = reviewed PAS clusters
```

每条 path 必须：

- 恰好选择一个 TSS alternative；
- 恰好选择一个 PAS alternative；
- internal choice 未经过时使用明确的 `not_traversed` 语义，而不是伪造空 bubble；
- 按真实转录方向得到有序 choice sequence。

终端 alternative representation 使用 terminal node 与首/末 adjacent-edge CIS states，不将 SOURCE/SINK 重新当成生物学 processing edges。

### Endpoint evidence/QC

在 terminal choices 进入模型前，必须冻结端点证据政策：

- TSS：5-prime completeness、annotation confidence、可用时的 CAGE/RAMPAGE 或独立 cap evidence；
- PAS：canonical poly(A) signal、internal-priming filter、3-prime completeness；
- cluster identity 不在模型运行时按距离静默生成；
- split-neutral endpoint identity 与 train-only support/eligibility 分开。

V2 最小版本优先只纳入高置信 terminals；若伪端点仍显著，再考虑概率 error model，而不是一开始增加新 trainable block。

## 7. 必须修改三：增强静态 CIS baseline

为避免 DNA/RNA dynamic branches 只是补偿弱 sequence baseline，新增固定、split-neutral、可解释的 CIS features：

- donor/acceptor splice-site strength；
- branchpoint 与 polypyrimidine tract；
- exon/intron length、GC 和局部结构性特征；
- promoter/core-TSS sequence context；
- PAS hexamer 与 downstream U/GU context。

这些首先作为确定性 features 进入 CIS/alternative representation。大型 sequence CNN/foundation encoder 只作外部 ablation 或预测上界，不作为主解释模型。

## 8. 必须修改四：构建 ChoiceGraph 与 pair catalog

从 processing graph 和 legal paths 派生第二张图：

```text
ChoiceGraph node = TSS / internal / PAS choice
ChoiceGraph edge = 按转录方向可共同出现在 legal path 的候选依赖
```

第一版只包含 path 上相邻的 next-choice pairs。direct TSS-PAS、TSS-远端 exon 和远距离 exon-exon 作为 skip pairs，只有相邻模型留下稳定 residual 后才审计准入。

新增至少以下字段/张量：

```text
pair_id
source_choice_id
target_choice_id
pair_type
fixed_transcript_order
path_pair_incidence[p, pair, source_alt, target_alt]
pair_identifiability
event_pair_routing
```

方向由 genomic/transcript order 固定，是生物学先验，不宣称由 cross-sectional molecule counts 学出了因果方向。

## 9. 必须修改五：pair 可辨识性与统计准入

pair interaction 必须先对 intercept、source unary、target unary 做残差化。对于 choices `c,d`：

\[
x_{cd}(p)=x_c(p)\otimes x_d(p).
\]

pair 只有同时满足以下条件才能训练：

- legal-path structural interaction rank 完整；
- train compatible-set supervision rank 完整；
- joint alternative combinations 和 informative molecule mass 足够；
- dynamic event gates 在 train 中有有效观测和变化；
- 在控制 State/unary effects 后，within-state/cross-fitted residual 仍支持 coupling；
- held-out fold 上有方向一致的 pair gain。

仅出现 `A1-B1` 和 `A2-B2`、没有交叉组合时，拒绝该 pair。第一版不拟合 partial-rank 子空间。

pair discovery/admission 使用 train folds；正式 pair effect 和 event attribution在 held-out folds 估计，并对预声明的 pair/event family 做 FDR 控制，避免 selection 后又在同一数据上报告过强置信度。

## 10. 必须修改六：Pairwise CRF 与 event-conditioned pair scorer

保留现有 unary `StateScorer` 与 `EventScorer`；新增静态和动态 pair potentials：

\[
\psi_i
=
\psi^{CIS}
+\psi^{State}_i
+\psi^{DNA}_i
+\psi^{RNA}_i.
\]

- `CIS pair`：稳定的组合偏好；
- `State pair`：共享细胞状态下的组合偏好；
- `DNA/RNA pair`：只归因给具名、局部路由的 events。

pair table 使用双零和 gauge：

\[
\bar\psi_{ab}
=
\psi_{ab}-\psi_{a\cdot}-\psi_{\cdot b}+\psi_{\cdot\cdot}.
\]

低秩 event-pair scorer 使用 event、source alternative、target alternative 三者的乘性交互；不能仅 concatenate 后接线性层，因为后者仍可分解为两个 unary effects。

所有 pair-side projections 零初始化，保证 `pair=0` 时逐值恢复 V1 unary path logits 和 NLL。

## 11. 必须修改七：重新定义 ATAC、DNA 和 RNA evidence

### ATAC/accessibility-only event

新增明确的 `promoter_accessibility` event：

- 不要求 TF motif；
- 贡献 TSS unary；
- 可按固定局部规则路由到相邻 downstream pair；
- generic openness 不归给碰巧命中的 TF。

### Factor-specific DNA event

继续使用：

```text
DNA motif × factor activity × accessibility × mapping reliability
```

它表示 factor-specific sequence-anchored evidence，不表示真实 occupancy。

### RNA event

继续使用：

```text
RNA motif / local pre-mRNA sequence × RBP activity
```

它可贡献 internal/PAS unary 或与 anchor choice 相邻的 admitted pairs。

### PAS 与 ATAC

PAS sequence/cleavage evidence是 primary local evidence。PAS-local ATAC 若使用，只作为 strand-aware、无预设正方向的 chromatin/elongation context；不假定 accessibility 越高 PAS usage 越高。

### Event routing

event 只能路由到其 anchor choice 的 unary 和 incident/admitted pairs，不能广播到全基因所有 pairs。gate 不复制，route 只引用同一个 event identity。

## 12. 必须修改八：定义 attribution semantics 与输出合同

mask 一个 event 时，必须同时去掉：

- 该 event 的全部 unary contribution；
- 该 event 的全部 pair routes/contributions；
- 然后重新计算 gene 内 path softmax、TSS/PAS marginals。

至少输出：

1. `terminal_choice_usage`；
2. `unary_event_contribution`；
3. `pair_event_contribution`；
4. `conditional_choice_contrast`；
5. `path_logit_decomposition`；
6. `counterfactual_path_probability_delta`。

对 event `j` 的条件作用报告：

\[
\Delta\operatorname{logit}_j(b:b'\mid a)
=C_{j,a,b}-C_{j,a,b'}.
\]

logit contributions 必须精确加和；probability effects 只能通过 masking + re-softmax 得到。所有输出保留 factor、motif、coordinate、cell state、gate、mapping reliability、choice/pair/path IDs 和 provenance。

## 13. 必须修改九：接入 high-DTU，但不改成 high-DTU-only task

### 师兄列表先做 provenance 审计

在看到实际文件前，它只能是 `external_high_dtu_candidate`。必须核查：

- cohort、species、reference/GTF/transcript annotation；
- cell states/conditions、donor/embryo replicate unit；
- DTU 方法、covariates、minimum support、effect size、FDR 和方向；
- transcript abundance 还是 conditional transcript usage；
- 与当前 cells/embryos/long-read labels 是否重叠；
- Ensembl ID 映射到 full7198 的一对一情况；
- 在 FABRIC 内的 choice、pair、event 和 molecule support。

使用规则：

- 独立 cohort 且 provenance 完整：外部 prior/challenge stratum；
- 使用当前 cohort 全部 labels：不得用于正式 train/val/test 选择，必须按 outer-train 重算；
- provenance 不清：仅 exploratory annotation。

外部列表、内部 train-DTU 和 held-out replication 必须是三个字段，不能混成一个 truth label。

### 内部 train-only DTU

每个 outer fold 内比较：

```text
M0: path usage ~ gene baseline
M1: path usage ~ gene baseline + cell state
```

继续使用 compatible-path likelihood，不强行给 ambiguous molecules 指派单一 transcript。冻结：

- `dynamic_replicated`；
- `stable_powered`；
- `indeterminate`。

stable 需要等效性证据和足够 power，不能由“不显著”定义。dynamic/stable matching 至少控制 expression、molecule mass、gene length、path/choice count、baseline entropy、event count 和 ATAC mapping quality。

DTU 分别计算 path、TSS、PAS、internal choice 和 pair residual；branch-specific high-information scope 为：

```text
high DTU
∩ identifiable choice/pair
∩ modality-specific event variability
∩ cross-embryo replication
```

## 14. 必须修改十：训练目标与 hierarchy

CIS/B0 保持 all-gene molecule objective。动态 branches 使用：

\[
\mathcal L_\lambda
=(1-\lambda)\mathcal L_{micro,all}
+\lambda\mathcal L_{macro,DTU}.
\]

其中每个基因先在内部按 molecule mass 平均，再由冻结的 train-only DTU weight 加权。所有 paired children 使用相同 folds、weights、seeds、parent checkpoint、early stopping 和 broad-scope degradation gate。

推荐 hierarchy：

```text
B0
└─ CIS-unary
   └─ CIS-joint                 + static pair
      └─ State-unary
         └─ State-joint         + state pair
            ├─ + ATAC/DNA unary+pair
            ├─ + RNA unary+pair
            └─ + ATAC/DNA+RNA unary+pair
```

Full 不继承已训练单模态 child；所有 modality children 从逐值相同 parent 启动。high-DTU hard subset 只作为 diagnostic upper bound，不作为正式总体。

## 15. 必须修改十一：观察偏差和 split

### Long-read observation bias

在相信 long-range TSS-PAS coupling 前，必须审计：

- transcript/path length 与完整捕获率；
- 5-prime/3-prime completeness；
- gene/path coverage 和 EC ambiguity；
- long-read chemistry/library batch。

若存在系统偏差，必须预先选择并冻结以下一种政策：固定 capture-bias offset、可比 path 限制、或 matched sensitivity analysis。该 observation correction 不进入 biological event attribution。

### Split

同时保留：

- 当前 embryo 内 cell holdout：同一 embryos 中新细胞的 transductive prediction；
- 9-fold leave-one-embryo-out：新 embryo 的机制复现/泛化。

每个 outer fold 的 DTU、pair admission、support、gate baseline、weights 和 threshold 全部只从 training embryos 得到。cells 不是独立 biological replicates。

## 16. 必须修改十二：评估、nulls 与生物学验证

### 评估 scopes

- `all eligible`：molecule-micro + gene-macro NLL；
- `train-defined dynamic challenge`；
- `matched stable control`；
- `modality opportunity`；
- `pair-identifiable`；
- TSS-exon、TSS-PAS、exon-exon、exon-PAS 分层。

每个 scope 报告 gene、choice、pair、path 和 molecule coverage。

### 必需 nulls

- stage/system/donor-matched factor-activity permutation；
- matched ATAC-neighbor/accessibility shuffle；
- route-preserving event identity permutation；
- pair-off ablation；
- matched random-gene controls，区分 DTU enrichment 与 gene balancing。

### 生物学验证

观察数据只支持 conditional coupling。强机制/因果语言需要 held-out perturbation：TF/RBP knockdown、CRISPRi/a、motif/site perturbation或相关 occupancy/elongation assay。

主文只选择 2–3 个预定义完整 cases：

- CTCF-PTPRC/CD45 exon 5：ATAC/motif/conditional inclusion/full paths；
- promoter/TSS 到 downstream path/PAS；
- RBP 到 exon/path。

每个 case 同时展示 locus graph、local ATAC、motif/factor activity、event logit、conditional choice contrast、complete paths、observed long-read usage和外部/扰动证据。ATAC openness 本身不能被写成 occupancy。

## 17. 代码与 artifact 修改边界

| 文件/对象 | V2 修改 |
|---|---|
| `docs/FABRIC_ARCHITECTURE_V2.md` | 新合同、方程、claims、split、DTU、pair admission |
| `configs/fabric_v2_*.yaml` | 固定 terminal/pair/DTU/model/evaluation policy |
| `graph.py` | 保留 processing graph；加强 terminal/path identity 和 observation QC |
| `choices.py` | terminal/internal choices、path choice sequence、unary identifiability；删除旧 terminal scope 语义 |
| `choice_graph.py` | ChoiceGraph、pair catalog/incidence、pair admission、event-pair routes |
| `motifs.py` | TSS/internal/PAS strand-aware regions；accessibility-only event geometry |
| `dataset.py` | 编译 terminal/pair/event/DTU tensors 与轴 identity/provenance |
| `model.py` | pair state/scorers、double centering、pair-aware path readout |
| `likelihood.py` | compatible-path likelihood 科学语义不变 |
| `train.py` | 新 hierarchy、mixed objective、zero-init parity、fold-local artifacts |
| `evaluate.py` | macro/DTU/pair scopes、conditional contrast、统一 masking 和 FDR outputs |
| `tests/` | terminal、pair rank、parity、leakage、routing、attribution、split tests |

新增核心 artifacts：

```text
choice_catalog.parquet
choice_alternatives.parquet
choice_pair_catalog.parquet
path_pair_incidence.parquet
choice_pair_identifiability.parquet
event_pair_routing.parquet
gene_dtu_scores.parquet
scope_metrics.tsv
event_unary_attribution.parquet
event_pair_attribution.parquet
path_logit_decomposition.parquet
```

旧 V1 PreparedDataset/checkpoints 直接判 incompatible 并重建，不写 adapter。

## 18. 关键测试

必须按依赖逐层通过：

1. 每条 path 恰好一个 TSS 和一个 PAS；
2. negative-strand choice order 正确；
3. 不再用 `full_length` giant choice 吸收依赖；
4. 2x2 全组合 pair rank PASS，对角-only FAIL；
5. within-state residual gate 拒绝纯 cell-state mixing coupling；
6. pair table 行列和为零；
7. pair=0 时逐值恢复 unary logits/NLL；
8. sparse pair incidence 与手算 path logit一致；
9. 同一 TSS 可促进一个 downstream alternative、抑制另一个；
10. event masking 同时清除全部 unary/pair routes；
11. ATAC-only 与 TF-motif-ATAC event 不混淆；
12. DTU/weights/pair admission 不读取 held-out labels；
13. children 使用逐值相同 parent；
14. total path logit 等于所有具名分量精确和；
15. compatible NLL brute-force parity 保持；
16. endpoint/internal-priming/length-bias fixtures 通过。

## 19. 推荐执行顺序与停止门

1. **审计师兄 high-DTU list 和当前 endpoint/observation bias**  
   provenance 或 endpoint evidence 不通过时，不进入正式 V2。

2. **不重训的 DTU dose-response 诊断**  
   若现有 DNA/RNA gain 不随 train-defined DTU 增强，停止“稳定基因稀释”解释，不做 DTU weighting。

3. **冻结 V2 contract、split 与 estimands**  
   在任何模型代码前确定 terminal、pair、DTU、masking 和 claim language。

4. **实现真实 terminal choices 与增强 unary baseline**  
   若端点 coverage/identifiability不足，收窄 TSS/PAS claims。

5. **实现 ChoiceGraph、pair incidence 与 rank gate**  
   pair=0 parity 不通过则禁止训练 joint model。

6. **实现 sparse pairwise CRF 与 event routes**  
   若 pair-identifiable held-out scope 无稳定增益，回退 unary FABRIC。

7. **实现 DTU-aware objective、embryo-LOO 与正式 evaluation**  
   若 shuffled modalities 与真实 modalities 相当，不能宣称 DNA/RNA regulatory evidence。

8. **扰动与 2–3 个预定义 cases**  
   只有 local conditional choice 和 full-path 两层都方向一致，才升级到强机制叙事。

## 20. 论文图谱

- **Figure 1–2**：ATAC atlas、cell-state accessibility、motif/regulatory landscape；
- **Figure 3**：TSS/internal/PAS usage、DTU 与 choice-coupling landscape，定义可解释任务；
- **Figure 4**：FABRIC V2 结构与 all-gene/high-DTU/stable/pair-identifiable prediction；
- **Figure 5**：factor/motif event → unary/pair conditional logit → complete paths；
- **Figure 6**：perturbation 与 CTCF/PTPRC、promoter→path/PAS、RBP→exon/path cases；
- **Supplement**：endpoint/capture QC、rank/support、permutation、threshold、seed/fold stability 和高容量上界。

## 21. Must / Should / Defer

### Must

- V2 contract；
- true terminal choices + endpoint QC；
- stronger static CIS sequence features；
- sparse choice graph + conditional/rank admission；
- double-centered unary/pair CRF；
- accessibility-only、DNA、RNA typed events；
- complete attribution contract；
- external list audit + train-only DTU；
- mixed objective + stable controls；
- embryo-LOO、micro/macro/scoped evaluation；
- observation-bias audit、FDR、perturbation/case evidence。

### Should

- adjacent pairs first，held-out residual 后再加 skip pairs；
- external CAGE/RAMPAGE/poly(A)/occupancy 作为正交验证；
- matched high-DTU/stable analysis；
- AR/path encoder 仅作为预测上界；
- 独立 external cohort 或 unseen-gene 附加泛化。

### Defer/Delete

- TF-RBP PPI/world model；
- dynamic all-gene GraphGPS；
- all-to-all learned adjacency；
- triplet/higher-order factors；
- path-ID embeddings/free coefficients；
- 把 PAS accessibility 设为固定正方向；
- high-DTU-only 正式任务；
- V1/V2 runtime compatibility infrastructure。

## 22. Change-My-Mind Evidence

- 若 DTU decile 与 DNA/RNA held-out gain 无关系，删除 DTU-aware weighting；
- 若 pair scope 无 held-out NLL 增益或 effects 跨 embryo 不稳定，删除 pair branch，保留 unary FABRIC；
- 若 terminal artifacts/coverage 无法控制，收窄到 internal processing，不做 TSS/PAS 强 claim；
- 若真实 modality permutation 与 observed modality 等效，收窄到 State/sequence prediction；
- 只有稳定三阶 residual 且数据 rank/power 足够时，才重新讨论 higher-order/path-history 模型。
