# FABRIC Architecture V2

**FABRIC: Factor-Aware Branch Regulation of Isoform Choice from DNA and RNA Motif Evidence**

> 一个以细胞特异、具名 DNA/RNA motif events 为输入，在 ONT matrix isoform structural paths 上建模相对使用概率，并将 event 影响连接到局部 alternative 与完整 path 的基因级图模型。

## 文档状态

| 字段 | 当前值 |
|---|---|
| 设计状态 | `V2_DESIGN_FINAL` |
| 日期 | `2026-08-14` |
| 实现状态 | `V2_UNIFIED_SINGLE_RUN_RUNTIME_IMPLEMENTED_REPROFILE_REQUIRED` |
| full-cohort 训练状态 | `NOT_AUTHORIZED_BY_THIS_DOCUMENT` |
| held-out test 状态 | `NOT_AUTHORIZED` |
| 后续 CIS 扩展 | AlphaGenome `embeddings_1bp`，`DEFERRED_CIS_EXTENSION`，当前禁用 |
| V1 历史合同 | `docs/FABRIC_ARCHITECTURE_V1.md` |
| 被取代的 V2 提案 | `refine-logs/FINAL_PROPOSAL.md` 中的 ChoiceGraph/pairwise-CRF 主线 |

本文档冻结 FABRIC V2 第一版的科学问题、模型方程、解释语义和实现边界。当前 `src/fabric` 已实现统一的单任务 V2 runtime、fixture validation 与真实数据构建路径；实现状态、真实 artifact admission、资源 profiling、训练启动授权和最终科学结果必须分别报告。本文件不授权 full-cohort 训练。V1 文档、代码与既有结果只作为历史身份保留；V2 不建立 V1/V2 双 runtime、checkpoint adapter 或兼容层。

## 1. 科学问题与 estimand

FABRIC V2 使用同一细胞的长读长监督、短读长 RNA context 和映射 ATAC context，回答：

> 哪个 factor 的哪个 DNA 或 RNA motif event，在什么细胞状态下，使哪个局部 RNA-processing alternative 的 matched-context 或 marginal relative log-mass 改变多少，并最终重新分配了哪些完整转录本路径的概率？

预测对象是细胞 \(i\)、基因 \(g\) 条件下，ONT matrix isoform structural-path 集合 \(\mathcal Y_g\) 上的相对概率分布：

\[
P_{i,g}(p),\qquad p\in\mathcal Y_g.
\]

这里的概率必须按当前观测过程解释为：

\[
P_{i,g}^{obs}(p)
=
P\!\left(
p\mid
\substack{
\text{来自 cell }i\text{、gene }g\text{，被长读长实验捕获并通过冻结的 pre-compatibility technical QC，}\\
\text{且其观测证据在冻结 ONT-matrix isoform catalog 下至少兼容一条合法 path 的 molecule}
}
\right).
\]

后文为简洁仍写作 \(P_{i,g}(p)\)。这里最后一个条件只表示 read/EC 的已接纳观测证据与至少一条 matrix isoform 结构相容，不证明真实 molecule 必然来自完整生物转录本总体；§3.3 必须单独审计被该条件排除的 molecules。当前 likelihood 不显式建模 transcript-specific capture、完整长度保留、mappability、PCR 或长度相关观测概率，因此它首先是 **matrix-catalog-compatible observed-library conditional isoform distribution**，不是未经校正的细胞内绝对 isoform abundance。只有在不同 paths 的观测概率可视为相同，或已有外部冻结的 path-specific observation calibration 时，才能把它进一步解释为 biological cellular isoform usage；该 calibration 不属于当前 V2。path readout 中的 \(\log(1+|p|)\) 是结构预测特征，不是 capture-bias offset。

模型不预测总基因表达量，不生成新的转录本结构，也不把 motif hit、factor expression 或 ATAC accessibility 等同于真实蛋白占据或因果调控。主要调控量是一个模型内反事实：删除或改变某项具名证据后，局部 alternative 的 matched-context/marginal relative log-mass 与 matrix-catalog-compatible observed-library path probability 如何变化。

## 2. 一句话方法

> **每个基因一张 processing graph；每个 processing token 由静态 CIS、细胞特异 DNA/ATAC events 和 RNA/RBP events 三块构成；三块经联合投影与固定 pre-normalization 后共同经过一次 gene-level GraphGPS 上下文化；每条 matrix isoform path 先汇总向量，再由一个共享的小型非线性 readout 产生 path logit；训练只使用 compatible-path likelihood。**

```text
matrix-matched per-gene processing graph + isoform paths
                         │
      ┌──────────────────┼──────────────────┐
      │                  │                  │
  static CIS       DNA/ATAC events     RNA/RBP events
      │                  │                  │
      └────────────[CIS | DNA | RNA]────────┘
                         │
             joint projection + fixed pre-LN
                         │
               one-layer gene GraphGPS
                         │
              contextual processing states
                         │
             matrix-path-specific vector pooling
                         │
               shared small path MLP
                         │
                 legal-path logits
                         │
               gene-level path softmax
                         │
             compatible-path likelihood
```

V2 的表达能力来自“细胞特异事件共同进图”和“完整路径上下文之后再标量化”。V2 不使用第二张 ChoiceGraph、pairwise CRF、显式 choice-pair potential 或 path-specific free parameters。

## 3. 固定生物学对象

### 3.1 Gene processing graph

每个基因只有一张 processing graph。生物学节点固定为 `TSS`、`donor`、`acceptor` 和 `PAS`，processing edge 固定为 annotation 中允许的 `EXON_CONTINUATION`、`SPLICE` 和 `RETAINED_INTRON` transition。所有 first/last、upstream/downstream、signed distance 和 endpoint role 均按转录方向定义。

为最小化对现有数据层的修改，模型 token 继续使用 processing edge。局部 `local_edge_index` 的定义完全沿用当前 `graph.py`：若两条 processing edges 在至少一条冻结的 matrix isoform path 上前后连续，则建立双向、去重的 line-graph adjacency；除此之外不额外连接所有“共享 processing site”但未在合法路径上连续的 edge pairs。非相邻的全基因关系由 GraphGPS 的 global attention 表达。V2 不把每条 transcript 单独建图，也不把 alternative 重新定义成第二张可学习图的节点。

### 3.2 ONT-matrix isoform paths

ONT matrix 的每条 transcript row 必须一对一映射到 matrix-matched GTF 中同一 gene graph 上一条冻结的、有序的合法 edge sequence。若多个 matrix transcript IDs 具有完全相同的 ordered edge sequence，它们折叠为一个 structural path，并保留 `transcript_aliases`；\(\mathcal Y_g\) 只包含 matrix rows 对应的唯一 structural paths，避免在 softmax 中重复计算相同结构。若 ordered edge sequence 不同，则即使共享 TSS/PAS 或大部分 edges，也仍是不同 paths。对当前已审计数据，101,067 个 matrix rows 没有 structural alias collision；17,706 个 primary genes 的完整模型轴合计 90,672 paths。定义 structural path-edge incidence：

primary structural candidate 必须位于单一 canonical nuclear chromosome/strand，且至少两条不同 structural paths 都是已解析的 ONT matrix rows、并分别在 train cells 中具有正 raw observation；GTF-only transcript 不得用于凑足两条。DTU score/label 在该 admission 冻结后才允许联结，不能改变 gene/path universe。当前共有 4,348 个恰好两条 matrix paths 的 canonical genes 满足该规则并进入 primary catalog。

\[
M_{p,e}=\mathbb 1[e\in p].
\]

union graph 只负责结构上下文化，不能作为新转录本生成器。运行时不得从 union edges 重新枚举 walks，也不得从 full GTF 加入 matrix axis 之外的 transcript。冻结的 matrix structural-path catalog 是唯一候选空间：\(\pi\) 的 softmax 分母、所有 \(\psi\) 聚合、compatible sets、NLL、evaluation 和 attribution 都只使用 \(\mathcal Y_g\)。矩阵外路径不参与计算，而不是作为零表达候选加入。

### 3.3 Long-read supervision

一个长读长 molecule 或 equivalence-class row \(k\) 可以只确定 compatible path set \(C_k\subseteq\mathcal Y_g\)，并携带 molecule weight \(n_k\)。模型不得把 ambiguous molecule 强行指派给单一 transcript。多个属于同一 `(cell, gene)` 的 compatible rows 必须共享同一组 path probabilities。

17,706-gene ONT-first catalog 所需的 compatible-EC artifact 是正式训练前的独立上游数据交付，不由模型 forward 或 trainer 临时生成。数据准备负责人必须在 `CompatibilityArtifactManifest` 中记录 producer、命令、代码版本、冻结的既有 ONT alignment identity（BAM/CRAM 或无损等价 alignment table）、reference、matrix transcript/crosswalk/structural-path catalog、cell/split manifest 和下述 QC/compatibility policy。producer 只从已冻结 alignments 提取 molecule evidence，不重新比对 FASTQ、不重建 BAM、不重新注释 GTF，也不发现新 path；对每个 molecule 保存 canonical cell/gene、split、pre-compatibility QC、ordered compatible structural-path IDs 与最终 fate，再仅将 `(cell, gene, identical C_k, fate)` 完全相同的 molecules 聚合为 EC row 和整数 \(n_k\)。train/validation 使用同一冻结规则生成；若新 10% test compatibility 尚未暴露，其 rules/producer 先冻结而 rows 延至 checkpoint 和报告规则锁定后生成，若早已暴露则必须如实记录 exposure，不能声称 blind。

该交付必须覆盖全部 17,706 个 structural candidates 并产生逐 split 的 molecule/count conservation、`LongReadCompatibilityAudit` 和输入 identity；没有 observation 的 gene 仍以零支持显式出现。只有 train 中 \(\sum_i w^{inf}_{i,g}>0\) 的 genes 进入实际 likelihood fit，其他保留为 graph-only/audit。历史 7,198-gene EC artifact 可以作为独立实现对照或重叠区 fixture，但不得静默补齐或定义新 catalog 的full-cohort 监督。artifact 未交付、任一 identity/QC/fate 不可复现、整数 mass 不守恒或 train/validation 使用不同规则时，implementation admission 直接失败；`src/fabric` 的职责是严格验证并消费该 artifact，而不是内置另一条兼容性生成 fallback。 `CompatibilityArtifactManifest.legal_path_catalog_identity` 必须与 `OntObservationProcessAudit.path_identity` 完全相同，并冻结 `model_isoform_universe=resolved_ont_matrix_structural_paths_only` 与 `matrix_structural_path_count=90,672`；任何不一致都阻止 training admission。

对于 \(|\mathcal Y_g|=2\) 的 gene，likelihood-informative compatible set 必然是 singleton；同时兼容两条 paths 的 full set 仍是 audit-only，不能进入 NLL 分母。

`RETAINED_INTRON` path 的正向 compatibility 需要单独冻结的 long-read evidence policy。只有 primary、非嵌合、通过预声明 MAPQ 与 alignment-anchor 阈值的 read，连续跨过目标 intron 两侧 exon–intron boundaries、在每侧满足冻结的最小 exon/intron aligned bases，且没有支持切除该 intron 的 splice junction，才标记为 `IR_alignment_supported`。只覆盖 intron 内部或单侧 boundary 的 read 不得据此把 retained-intron path 加入其 positive compatible set；该项标记为 `IR_evidence_censored`，并只用剩余已接纳的 junction/end evidence 重新构建 \(C_k\)。若没有剩余区分证据，则令 \(C_k=\mathcal Y_g\) 并按下文作为 audit-only row；不得因忽略不充分的 intronic alignment 而反向把 spliced paths 当作已获支持。

另设与 §3.4 `PathIdentifiabilityIndex` 正交的 `IR_biogenesis_context = processed_context_supported | mature_vs_nascent_unresolved`。`processed_context_supported` 第一版定义为同一 molecule 还含至少一个其他通过同一 QC 的 canonical splice junction；这只能证明 transcript 已发生部分 processing，不能单独证明成熟。protocol-specific mature/full-length/end support、poly(A) evidence、internal-priming flag 与 genomic-DNA contamination flag 分开保存，不合并成一个匿名 quality score。`IR_alignment_supported AND processed_context_supported` 的 molecule 可进入主要“processed-context RI-compatible path”解释；仅通过 alignment 者标记为 `mature_vs_nascent_unresolved`，可以保留其诚实的 compatible-set监督，但不得声称成熟 retained-intron isoform。两外显子或其他结构可以在 matrix paths 间监督可辨识，同时在成熟 RI 与未剪接前体之间仍不确定；不得把这种 biogenesis uncertainty 错记为 supervision-unidentifiable，也不由模型补猜。只有另有冻结的 protocol-specific mature-transcript evidence 时，才能把措辞从 processed-context 升级为 mature RI。

上述阈值与 flags 必须在 split 和 path outcome 前冻结。每个 library/donor 至少报告 IR 双边界支持、`IR_evidence_censored`、multi-intron unspliced pattern、internal priming、genomic-DNA contamination 与两类 `IR_biogenesis_context` 的 molecule 比例；稀疏单细胞不强制逐 cell 估计阈值。降解或截短首先造成证据缺失和 compatible-set 变宽，不能仅因 read 未覆盖某个 splice junction 就把它判为 intron retention。V2 不因此增加 nascent-RNA latent state、IR-specific prediction head 或降解生成模型。

所有训练、gate weighting、evaluation normalization 和 state residual 统一使用 likelihood-informative row set：

\[
\mathcal K^{inf}_{i,g}
=
\left\{
k:n_k>0,\;
\varnothing\ne C_k\subsetneq\mathcal Y_g
\right\}.
\]

\(C_k=\varnothing\) 不得再笼统并入 technical QC failure。pre-compatibility technical QC 在运行 matrix-catalog compatibility operator 之前完成；通过该 QC、具有正 molecule mass 但得到空集的 row 标记为 `no_matrix_isoform_compatible` 并从 likelihood 排除。这个标记只说明当前证据不兼容冻结 catalog，不能单凭它宣称 novel isoform。\(C_k=\mathcal Y_g\) 则是 matrix-catalog-compatible 但对 path 选择无信息的 row，其 compatible likelihood 恒为 \(-\log1=0\)，只保留在 observation audit，不进入 informative molecule mass、loss numerator/denominator、support 或 residual。不得在不同模块中为 `informative` 使用不同定义。

数据层必须生成 reporting-only 的 `LongReadCompatibilityAudit`。固定 waterfall 为 `captured/gene-assigned -> pre-compatibility technical-QC pass -> no_matrix_isoform_compatible (C_k=empty) | likelihood_informative (nonempty proper subset C_k) | matrix_catalog_compatible_uninformative (C_k=Y_g)`。compatibility fraction 的统计总体固定为：已分配到 model-admitted gene、该 gene 具有非空冻结 catalog \(\mathcal Y_g\)、通过 pre-compatibility technical QC 且 molecule mass 为正的 rows；末端三类在该总体内互斥且完备。audit 同时报 row count 与 molecule mass，compatibility fraction 以该总体的总 molecule mass 为分母；分母为零的 stratum 标记 `not_estimable`，不得填零。每个 stratum 的三类末端 molecule mass 之和必须精确复现其 technical-QC-pass 总 mass（允许固定数值容差），row count 与 molecule mass 分开报告，至少按 split、library/donor、gene 和预声明 reporting cell state 分层。technical-QC failure 与 `no_matrix_isoform_compatible` 使用不同 reason code；该 audit 不得用 held-out enrichment 反向修改 matrix structural-path catalog、兼容规则、gene admission 或 case selection。若 matrix-catalog-incompatible fraction 在特定状态或基因中富集，相应 path-distribution claim 必须明确带上述条件化边界。

对固定 gene \(g\)，记 \(\mathcal K^{inf}_{train,g}=\bigcup_{i:(i,g)\in train}\mathcal K^{inf}_{i,g}\)。

### 3.4 Supervision identifiability

compatible-path likelihood 不会凭空产生 transcript-level information。令 train supervision 的 informative EC–path compatibility matrix 为：

\[
B^{train}_{k,p}=\mathbb 1[p\in C_k],
\qquad
k\in\mathcal K^{inf}_{train,g}.
\]

首先按 train \(\mathcal K^{inf}\) 中的不同 compatible-set pattern 建立矩阵；重复 rows 只合并权重，不重复改变 rank。若两个合法 paths 的 train compatibility columns 完全相同，则定义 \(p\sim_g q\)。由此得到 observational-equivalence groups \(E_1,\ldots,E_m\)，并定义折叠矩阵 \(\widetilde B^{train}\in\{0,1\}^{K\times m}\)。每个 group 的总概率为：

\[
P_{i,g}(E)=\sum_{p\in E}P_{i,g}(p).
\]

train catalog 的 group contrasts 只在下式成立时记为 `cohort_contrast_separable`：

\[
\ker(\widetilde B^{train})
\cap
\left\{z:\mathbf 1^\top z=0\right\}
=\{0\},
\]

等价地，

\[
\operatorname{rank}
\begin{bmatrix}
\widetilde B^{train}\\
\mathbf 1^\top
\end{bmatrix}
=m.
\]

数值 rank 使用 SVD，阈值固定为 \(10^{-8}\sigma_{max}\)；若 \(\sigma_{max}=0\)，rank 为零。另要求每个 group 至少有一个 group-specific supported molecule equivalent：

\[
S_h^{train}
=
\sum_{k\in\mathcal K^{inf}_{train,g}} n_k\,\mathbb 1[C_k\subseteq E_h]
\ge 1.
\]

这一 molecule threshold 是保守的 direct-support policy，不是上述 rank 条件的数学必要条件。这只是 **train catalog 层面的 contrast separability**，不能由跨细胞 pooled rows 推导任一细胞的 path usage 已被直接观测。

对每个 held-out \((i,g)\)，先在原始 path-level \(B_{i,g}\) 上检查每一 row 是否对每个冻结 train group \(E_h\) 都为全零或全一；只有这种 group-constant row 才能折叠成 \(\widetilde B_{i,g}\) 并进入 cell-level augmented-rank 与 \(S_{i,h}\ge1\) 检查。若 held-out 新 EC row 只覆盖某个 \(E_h\) 的部分成员，它仍进入原始 compatible likelihood，但记为 `novel_split_group_row`，不得用 test evidence 重定义 train groups 或升级 support tier。通过 cell-level 检查者标为 `direct_cell_supported`；仅 train catalog 通过者标为 `cohort_identifiable_model_prediction`；train catalog 未通过者标为 `supervision_unidentifiable_prediction`。V2 第一版不以跨细胞 pooled rank 冒充 cell-specific identifiability，也不增加 Fisher/Jacobian-based 参数可辨识性声明。

`PathIdentifiabilityIndex` 必须只用 train EC patterns、compatible sets 和 molecule support 构建，再冻结到 validation/test；explanation manifest 也不能依据 held-out support tier 或 novel resolving rows 选择 cases。只有属于 singleton group、train catalog contrasts 通过且当前 cell 为 `direct_cell_supported` 时，才允许称“该细胞的 transcript-usage target 有直接长读长区分支持”；event attribution 本身在所有 tiers 中仍是 model-derived counterfactual，不能称为长读长直接证明。若 \(|E|>1\)，网络仍可为组内 paths 产生不同预测，但这些差异必须标记为 `model_resolved_within_supervision_unidentifiable_group`，主要报告 group probability 和 counterfactual change，不得冒充 transcript-specific evidence。若一个 equivalence group 跨越待比较 alternatives 或 matched-context partitions，则对应 local contrast 不可报告。

## 4. 静态 CIS block

静态 CIS block 表示细胞不变的序列与结构先验。V2 主模型不训练新的 sequence CNN，只使用以下固定、split-neutral、可审计的字段组：

1. structure：processing-edge type、source/destination site-type one-hot；
2. geometry/annotation：\(\log(1+\text{span bp})\)、\(\log(1+\text{length bp})\)、transcript-oriented relative position、annotation confidence、现有 edge prior score；
3. local sequence：edge GC fraction；
4. splice competence：适用 endpoint 的 donor strength、acceptor strength、branchpoint score 与 polypyrimidine-tract score；
5. end processing：TSS/core-promoter score、poly(A)-hexamer score、PAS downstream U/GU fraction；
6. 每个仅对部分 edge/site 适用的 sequence score 对应一个显式 availability mask。

不适用的值置零但 mask 为零；真实数值零与不适用不得混淆。具体 scanner/version、reference build、strand convention、sequence windows、fixed transforms、输出顺序及 normalization 必须写入一个 `CISFeatureManifest` 并在实现前冻结。连续 CIS 特征的 normalization population 固定为 train-admitted gene universe 的 unique structural-edge catalog：每个 `edge_id` 恰好计一次，不得因其出现在多个 cells、molecules、paths 或 transcripts 中而重复加权。每一项连续特征只在对应 availability mask 为一的 edges 上拟合均值与尺度；one-hot 与 availability masks 不标准化；标准化完成后，不适用项仍置零。若某连续特征在该 catalog 上低于固定 numerical tolerance，则整列记为 `constant_cis_feature` 并从所有 splits 的模型输入中排除，不得用 epsilon 制造虚假变化。所有统计量只从 train-admitted structural catalog 拟合并冻结到 validation/test，不能使用 cell/molecule weights 或 held-out outcomes。上述字段不得根据 train/validation/test outcome 选择，也不得在未修订本合同的情况下追加匿名 sequence feature。当前 V2 主模型以这套显式特征作为唯一 CIS 输入。

### 4.1 Deferred AlphaGenome CIS extension

AlphaGenome `embeddings_1bp` 被保留为明确的后续设计 `DEFERRED_CIS_EXTENSION`。启用后，它应作为冻结的、细胞不变的静态 CIS 表示，与上述显式 CIS 特征拼接，而不是替代 DNA/ATAC 或 RNA/RBP 动态通道：

\[
c^{extended}_e
=
\operatorname{concat}
\left[
c^{explicit}_e,\,
c^{AlphaGenome}_e
\right].
\]

第一版扩展只允许离线提取并冻结 AlphaGenome 表征，不微调 AlphaGenome。启用前必须另行冻结 model/checkpoint 与 reference build、GTF 和 strand/coordinate convention、每类 TSS/donor/acceptor/PAS 的 transcript-oriented flank、`embeddings_1bp` 到 processing-edge token 的 pooling、输出维度与缓存格式。当前文档不预先指定 16 bp 或其他 flank，因为一碱基 embedding 已包含多大上下文、服务器实现返回哪一层表示尚未形成数据合同。

正式比较至少包括 `explicit CIS`、`explicit CIS + AlphaGenome` 和必要时的 `AlphaGenome-only` 诊断。该扩展的所有条件都实例化同一个固定宽度的 \([explicit\ CIS\mid AlphaGenome]\) 输入和完全相同的 CIS projection/GraphGPS/path-MLP 参数量；缺失 block 在 projection 前置零，因此加入 AlphaGenome 不能通过增加 trainable parameter count 获得不公平优势。所有动态模型比较必须保持同一 CIS 版本。未经独立修订与消融，AlphaGenome 不能通过配置开关静默进入 V2 主结果。

## 5. MotifEventCatalog

### 5.1 Event identity

每个调控候选必须是可独立审计和干预的具名物理 event。数据层分成两张冻结表，不能继续把 `choice_id` 或 alternative relation 编入物理 `event_id`。下述字段、identity keys 与 collapse/routing 算法属于 V2 实现 schema，必须逐项实例化；“至少保存”只允许增加不参与 identity、模型输入、筛选、归因 estimand 或 claim 的纯 provenance/QC 字段，不能借此增加新的科学语义。

`PhysicalEventTable` 的主键是 `event_id`，至少保存：

```text
event_id
target_gene_id
factor_entity_id or NULL
factor_identity_kind = unique | factor_equivalence_group | accessibility_only
cap_evidence_class = motif_anchored | accessibility_only
candidate_factor_ids
activity_entity_id or NULL
activity_gene_ids
activity_proxy_rule or NA
modality = DNA | RNA
event_kind or NA
motif_id or NULL
motif_equivalence_family_id or NULL
source_motif_ids
chromosome
start/end/strand
source_hit_coordinates, when collapsed
motif_score
orientation
peak_id, when applicable
peak_support, when applicable
gate_key_id or NULL
is_self_factor = true | false | NA
source_valid
has_retained_route
gate_key_active
model_active
admission_reasons
```

DNA/RNA motif hit 的物理 identity key 固定为 `(target_gene_id, modality, factor_entity_id, motif_equivalence_family_id, chromosome, canonical_start, canonical_end, strand, peak_id-or-NA)`；表中的 `start/end` 即 collapse 后的 canonical interval，accessibility-only event 使用 `(target_gene_id, DNA, accessibility_only, chromosome, start, end, strand, peak_id)`。`event_kind` 只是在 cap 完成后由 final retained routes 汇总得到的 reporting-only 字段，不属于物理 identity key，也不进入模型特征；允许值固定为 `TSS_PROXIMAL | SPLICE_SITE_PROXIMAL | EXON_CONTAINED | DNA_INTRAGENIC | PAS_PROXIMAL | MULTI_ANCHOR`。一个 event 的 final retained routes 若跨越多个单一类别，则 `event_kind=MULTI_ANCHOR`，同时保存完整 route-level anchor/region 类型；event 是否为 accessibility-only 已由 `factor_identity_kind` 唯一表达，不再复用 `event_kind`。同一 genomic source hit 被两个重叠 scan windows 命中时必须先形成一个 physical event、再产生两条 routes，不能因 window/anchor 不同复制 event identity。物理 catalog 使用 route-independent 的暂定 identity 完成 collapse；全部候选 routes 生成后再以最终 canonical interval 和上述 key 生成稳定 `event_id`，因此不存在 event ID 依赖尚未生成 routes 的循环。

`motif_equivalence_family_id` 只合并同一 factor/entity 内由预先声明为等价的 PWMs 产生、且 genomic footprints 按冻结 overlap rule 表示同一物理结合位点的 hits；不同 factor entities、非等价 binding modes、不同不重叠 loci 或不同 peak identities 不得合并。每个未合并 motif 以其自身 `motif_id` 作为 singleton family ID。family membership、PWM-similarity/metadata rule、overlap predicate/tolerance、canonical interval/representative motif 与 score aggregation 必须在查看 path outcome 前冻结并写入 catalog manifest。collapse 必须发生在 route 生成之前：在每个 `(target_gene_id, modality, factor_entity_id, motif_equivalence_family_id, strand, peak_id-or-NA)` bucket 内，将满足该 manifest 中精确定义的 overlap predicate（例如 overlap bp 或 reciprocal-overlap threshold；不能只写“相近”）的 source-hit intervals 构成无向图，并以 connected components 定义 clusters；因此链式重叠属于同一 cluster，不允许实现改成 greedy NMS 或 complete linkage。每个 component 再由最高可用 source-normalized/catalog-calibrated motif quality、source priority、较小 start/end 与 motif ID 的固定字典序 tie-break 选择 representative hit，其 interval 为 canonical interval、其 `motif_id` 为 representative motif；若 calibration 不可用，直接从 source-local rank 开始上述 tie-break。合并后保留全部 `source_motif_ids` 与 `source_hit_coordinates`，不允许多个模型不可区分的等价 PWM hits 重复累加 gate。若 motif source 没有足够信息建立等价 family，则不同 `motif_id` 默认保持不同，不得凭共同 factor 名称粗暴合并。

这里的两个“equivalence”不可混淆：`factor_identity_kind=factor_equivalence_group` 表示一个 motif 无法区分多个候选 proteins，因此限制 factor claim；`motif_equivalence_family_id` 表示同一 factor/entity 的多个等价 PWMs，因此用于避免同一物理位点重复计数。前者改变 activity/claim entity，后者只改变 motif-hit deduplication。

identity 字段关系必须唯一。对 `unique` event，`factor_entity_id=activity_entity_id` 且 `candidate_factor_ids=[factor_entity_id]`；对 `factor_equivalence_group` event，`factor_entity_id=activity_entity_id` 均为冻结的 group ID，`candidate_factor_ids` 保存全部候选 proteins；对 `accessibility_only` event，`factor_entity_id=activity_entity_id=NULL`、`candidate_factor_ids=[]`、`activity_gene_ids=[]`、`activity_proxy_rule=NA`、`motif_id=motif_equivalence_family_id=NULL`、`source_motif_ids=[]`、`orientation=NA`、`is_self_factor=NA`。对 motif event，`motif_id` 始终等于该 physical event 的 representative motif ID；singleton event 的 `source_motif_ids=[motif_id]`，collapsed event 的 `source_motif_ids` 保存全部成员且包含 `motif_id`。因此不另设 `factor_group_id` 或 `representative_motif_id` 别名字段。

`gate_key_id` 是 §6.2 中冻结 gate-key tuple 的稳定 ID；accessibility-only event 也必须引用其 \((g,p)\) gate key。event 的最终激活规则唯一固定为 `model_active = source_valid AND has_retained_route AND gate_key_active`。`admission_reasons` 是按 `invalid_source`、`no_retained_route`、`inactive_gate_key` 固定顺序保存的全部失败原因列表，不能在多项同时失败时任意挑一个；gate-key 内部失败原因另在 `GateAdmissionManifest` 中完整保存。没有 retained route 的 event 令 `event_kind=NA`。

`EventRouteTable` 的主键为 `(event_id, anchor_region_id, anchor_site_id-or-NA, edge_id)`，至少保存：

```text
event_id
anchor_region_id
anchor_site_id or NA
edge_id
route_weight
region_type
anchor_type
transcript-oriented side
signed_distance_bp
edge_relative_position
distance_to_5prime_boundary_bp
distance_to_3prime_boundary_bp
geometry_kind = site_window | edge_contained
```

多个 motif hits 不得在进入模型前压缩成一个只保留 best hit 的匿名 site-factor token。若 motif 本身不能区分同一家族的多个 proteins，则 event identity 必须是预先冻结的 `factor_equivalence_group`，不能根据 outcome 或当前细胞表达选择一个“owner factor”，归因也只能报告到该 group。`activity_entity_id` 对 unique factor 等于其 factor ID，对 `factor_equivalence_group` 等于冻结的 group ID；`activity_gene_ids` 保存构成 activity proxy 的全部源基因。group membership 与 activity proxy 规则必须在看见 outcome 前冻结。

静态候选截断不能再按 choice/alternative 分组。每个 physical event 先由 identity 字段确定一个不进入模型特征的派生 `cap_evidence_class`：`unique` 与 `factor_equivalence_group` 均为 `motif_anchored`，`accessibility_only` 单独为 `accessibility_only`；RNA events 恒为 `motif_anchored`。先按上述 `motif_equivalence_family` 与物理 identity 规则去重并生成 routes，再在每条 route 的 `(target_gene_id, modality, cap_evidence_class, anchor_region_id, region_type, anchor_type)` 内按固定静态质量排序，每组最多保留 16 个 physical events；不得使用 route 汇总后才产生的 physical `event_kind` 作为 cap key。这样 motif-anchored DNA 与 generic accessibility context 不使用不可比的排序键相互驱逐，同时 unique factor 与 factor-equivalence-group events 仍在同一 motif bucket 中竞争，不能因 factor identity 模糊而额外获得一份配额。

同一 event 即使在多个 anchor groups 中入选仍只有一个 event row。若部分 routes 因 cap 未入选，只保留入选 routes并对该 event 的 route weights 重新归一化；随后仅由这些 final retained routes 生成 reporting-only `event_kind`。motif-anchored DNA 排序依次使用 source-normalized/catalog-calibrated motif quality、peak support、geometry-specific proximity 与 event identity；RNA motif-anchored 使用 source-normalized/catalog-calibrated motif quality、geometry-specific proximity 与 event identity；accessibility-only 始终使用 peak support、geometry-specific proximity 与 event identity。该 motif quality 只需在静态 catalog 内提供确定性排序；它与 §5.5 中是否把 motif score 作为可比较数值特征是两个独立决定。只有 motif-anchored event 在 calibration 不可用时使用 source-local rank、source priority、geometry-specific proximity 和 event identity 的冻结 fallback，与上述 representative 选择一致；accessibility-only 不存在 motif source-local rank，也不得进入该 fallback。site-window route 的 proximity 是较小 \(|\text{signed distance}|\)；edge-contained route 的 proximity 固定为较小的 \(\min(d_{5'},d_{3'})\)，不允许为无 site anchor 的 route 臆造 signed distance。每个含 `cap_evidence_class` 的 bucket 都保存 candidate count、selected count、cap saturation、`motif_equivalence_family` collapse count 和 boundary quality。该截断不能读取细胞 outcome。

上述 16-event cap 是 **per `(anchor group, cap_evidence_class)` bucket** 的 catalog/计算边界，不是整个 anchor group 或 per processing-edge token 的总量上限；DNA anchor 同时存在两类证据时可分别保留至多 16 个 motif-anchored 与 16 个 accessibility-only events，即该 anchor 最多 32 个，RNA anchor 只有 motif-anchored bucket。来自多个 anchor groups 的 routes 还可以汇入同一 token。route audit 必须显式分成两个总体：`catalog_burden` 对 cap 后全部 retained routes 保存 distinct physical-event 数、distinct anchor-group 数、saturated anchor-group 数及 route \(L_1\) mass；`model_input_burden` 只对 `model_active` routes 按 `(target_gene_id, modality, edge_token_id)` 保存 distinct physical-event 数、distinct active gate-key 数、distinct anchor-group 数、route \(L_1\) mass 与下式 \(B^{gate}\)。为识别多个 model-active events 共享同一 gate key 所造成的结构性集中，定义

\[
B^{gate}_{g,M,e}
=
\sqrt{
\sum_{\tau}
\left(
\sum_{r\in\mathcal R^M:\,\tau(j(r))=\tau}
R^M_{r,e}
\right)^2
}.
\]

这里的 \(B^{gate}\) 是 model-input routing burden diagnostic，不进入模型输入或样本权重。它区分“由不同 gate keys 承载”与“多个 events 共享同一 gate key”的结构，但不估计不同 gate keys 之间的协方差，也不等于动态 block 的真实方差；因此不能把它或 motif catalog 密度直接解释为生物学剂量。

此外必须生成 reporting-only 的 `RouteDegreeCapAudit`，把 route-record multiplicity、distinct edge degree、anchor 数量与 cap 后重归一化分开记录。其 split-neutral `catalog` population 是 source validation 与 biological physical collapse 后、cap 前至少有一条候选 route 的 events；另生成只保留 `model_active=true` events 的 train-derived `model_input` view，但不得重算或更换任何 structural denominator。对 event \(j\)，令 \(D_j^{pre}\) 与 \(D_j^{post}\) 分别是 cap 前候选 routes 和 cap 后 retained routes 的总数；对其 anchor region \(a\)，令 \(n_{j,a}^{pre/post}\) 是相应的 per-anchor route-record 数，\(d_{j,a}^{pre/post}\) 是 distinct incident-edge 数，\(A_j^{pre/post}\) 是 distinct anchor-region 数。另保存每个 edge 收到的该 event route 数，以发现多个 anchors 汇入同一 token。当前等权 production routing 的 cap 前 reporting reference 与 cap 后实际权重分别为：

\[
R_{j,r}^{pre}=\frac{1}{D_j^{pre}},
\qquad
R_{j,r}^{post}=\frac{1}{D_j^{post}}
\quad (r\text{ retained}),
\]

因此 audit 必须显式保存：

\[
m_{j,a}^{pre}=\frac{n_{j,a}^{pre}}{D_j^{pre}},
\qquad
m_{j,a}^{rawret}=\frac{n_{j,a}^{post}}{D_j^{pre}},
\qquad
m_{j,a}^{post}=\frac{n_{j,a}^{post}}{D_j^{post}},
\qquad
L_j^{drop}=1-\frac{D_j^{post}}{D_j^{pre}},
\qquad
\kappa_j^{renorm}=\frac{D_j^{pre}}{D_j^{post}}.
\]

其中 \(m_{j,a}^{pre}\) 是 cap 前 anchor mass，\(m_{j,a}^{rawret}\) 是保留 routes 在未重归一化 reference 下的 mass，\(m_{j,a}^{post}\) 是 production mass，\(L_j^{drop}\) 是 dropped mass，\(\kappa_j^{renorm}\) 是每条 surviving route 相对 cap 前的确定性放大倍数。逐 anchor 还必须保存 `cap_loss = m_pre - m_rawret`、`renorm_gain = m_post - m_rawret`，并验证 `m_post - m_pre = renorm_gain - cap_loss`。表中至少保存 `audit_population`、`event_id`、target gene、modality、single/multi-anchor 标记、\(D_j^{pre/post}\)、\(A_j^{pre/post}\)、逐 anchor 的 \(n_{j,a}^{pre/post}\)、\(d_{j,a}^{pre/post}\) 与三种 mass、per-edge route-record counts、`dropped_route_mass`、`renormalization_factor`、发生 route 删除的 cap bucket/reason，以及 `external_only_coupling` 标记；后者严格指该 anchor 自身未丢 route、却因其他 anchor 丢 route 而增加 post-cap mass。这个 cross-anchor renormalization 是 catalog coupling，不得隐藏在 token burden 总量中，也不得称为生物学剂量变化。本表的 ratio equations 只对上述 \(D_j^{pre}>0\) population 定义；若另在完整 `PhysicalEventTable` 中联结零 candidate-route event，则标为 `not_applicable_no_candidate_route`，全部 mass/ratio quantities 为 `NA`。当 \(D_j^{pre}>0,D_j^{post}=0\) 时，\(m^{rawret}=0\)、\(L^{drop}=1\)，只有 \(m^{post}\)、\(\kappa^{renorm}\) 与 `renorm_gain` 为 `NA`，且该 event 不得 `model_active`。

该 audit 不改变本版的 sum-to-one production routing，也不提供 inference-time broadcast、\(1/\sqrt d\) correction 或 site-carrier 开关。§16.2 和 §17 必须用 single-anchor 与 multi-anchor synthetic fixtures 分别检查 route degree 和 cross-anchor cap sensitivity；一般情况下 edge \(e\) 收到的是 \(n_{j,e}/D_j\)，只有 one-route-per-distinct-edge 的审计 fixture 才退化为 \(1/d\)。该输入层缩放只是当前方程的代数实现校验，不自身构成失败。真正的 capability gate 必须基于 degree/cap 以外保持相同、且真值预设为 degree-invariant 的 planted local/path effect recovery、gradient 与 matched-context \(\Delta\rho\)，其 failure tolerance 只可由 synthetic/train reference 在读取 validation/test 结果前冻结。若该 gate 失败，当前 routing 不能获得 training admission；任何 broadcast、degree correction 或 site carrier 都需要另行修订合同并从头训练，不能根据 held-out test 结果选择。

染色体绝对坐标、`event_id`、`peak_id`、`motif_id`、`motif_equivalence_family_id`、target gene ID 和 transcript ID 只用于 provenance、去重、对齐、routing 与报告，不作为模型数值特征。

### 5.2 Accessibility-only DNA event

generic accessibility 与 factor-specific DNA evidence 必须分开。accessibility-only event 主要用于 promoter/TSS，也允许作为 intragenic 或 PAS 周围的无预设方向 chromatin context。此类 event 的 `factor_entity_id=NULL`、`motif_id=NULL`，不能归因给碰巧在该区域出现的 TF motif，也不能预设 accessibility 对 PAS usage 的作用方向。

### 5.3 Factor-specific DNA event

factor-specific DNA event 表示以下观测证据的组合：

```text
TF identity
× DNA motif hit
× current-cell factor activity
× mapped ATAC accessibility
```

它表示 factor-specific、sequence-anchored、accessibility-supported association，不表示真实 occupancy。ATAC mapping support 作为单独诊断报告，不乘入该 event 的动态幅度。DNA motif library、factor mapping、reference build、peak universe 和坐标约定必须在数据边界冻结。每个 factor-specific DNA source hit 在 collapse 前必须唯一映射到一个 `peak_id`；若冻结 peak universe 存在重叠 peaks，manifest 必须用 overlap、peak support 与稳定 peak-ID tie-break 给出唯一归属，同一 source hit 不得因重叠 peaks 复制成多个 physical events。

### 5.4 RNA/RBP event

RNA event 表示：

```text
RBP identity
× RNA motif hit in an explicit pre-mRNA window
× current-cell RBP activity proxy
```

RNA motif window、strand、pre-mRNA sequence 与 processing anchor 必须显式。V2 不从 RNA/ATAC/sequence 推断 WT1-U2AF65、SPI1-NONO 等蛋白物理互作，也不把模型中的统计交互解释为 PPI。

第一版有意不扫描完整长 intron，因此必须生成 reporting-only 的 `RNAWindowCoverageAudit`。其分母先限制为 reference-build 可唯一映射、属于 eligible gene 且 factor/entity 可映射的冻结 reference sites；waterfall 固定为 `eligible frozen reference sites -> inside allowed RNA windows -> has legal route -> retained after cap -> model_active`。前四步是 split-neutral structural/catalog coverage；`model_active` 后缀依赖 train-only gate admission，只能在 train 拟合后生成并冻结到 validation/test，不能把整个 waterfall 称为 split-neutral。

区域分类使用全部 matrix isoforms 与固定优先级：只要对任一合法 transcript 为 exonic 即归 `exonic`；否则若落入任一允许的 donor/acceptor RNA site-window 则归 `splice_proximal_intronic`；否则若落入其他允许的 TSS/PAS site-window 则归 `other_allowed_site_proximal`；只有对该 gene 全部合法 transcripts 均非 exonic 且不落入任何允许 RNA site-window 时才归 `deep_intronic`。audit 至少按 factor/entity 与这四类分层。若分母来自冻结的 eCLIP 或其他外部实验支持位点，只能称为“reference experimentally supported site coverage”，并记录 assay、biosample/cell-context 与 reference-build identity；若分母只是全内含子 motif scan，则只能称为“motif-candidate coverage”，不得称为已知 RBP 结合位点覆盖率。该 audit 不为生成分母而把所有长内含子 motif hits 加入模型，也不改变第一版 RNA window。

### 5.5 Fixed event feature vector

进入模型的 fixed feature 分为 event、route 与一个窄的 factor-conditioned context block。令 \(f_j\) 为 resolved factor 或 `factor_equivalence_group` one-hot；accessibility-only event 使用独立 `OPEN_ONLY` 类别。\(u^{event}_j\) 只包含 \(f_j\)、同一 motif/PWM family 内固定校准的 motif score、orientation categorical field，以及 DNA event 的 \(\log(1+\text{peak support})\)；reporting-only `event_kind` 不进入该向量。未经 family/PWM 内校准的不同 motif raw scores 不得直接当作跨 motif 可比较的结合强度；它们只能用于各自 catalog 内的静态 rank。若当前 motif source 无法提供在 train-independent sequence calibration set 上冻结的 family/PWM 内 score transform，则 motif score 不进入 \(u^{event}_j\)，只参与候选排序与 provenance；不得用当前 cohort path outcome 校准。route record \(r\) 的 \(u^{route}_r\) 包含 §7 定义的 geometry-kind-specific 连续几何，以及 geometry-kind、region、anchor、side 与 availability categorical fields；这些 categorical fields 的实际 reference coding 在下文冻结。

为避免同一 token 内“factor A 在位置 1、factor B 在位置 2”与交换后的配置在投影前发生线性聚合碰撞，另定义预声明的低维机制上下文 \(q_r\)：RNA 只包含 exon/intron region、transcript-oriented `UPSTREAM`/`DOWNSTREAM`/`OVERLAP_ANCHOR`、donor/acceptor/TSS/PAS anchor role 和固定 distance bins；DNA 只包含 promoter/intragenic/PAS region、anchor role 和固定 distance bins。orientation 只有在预声明的 DNA strand/orientation 机制假设下才进入 \(q_r\)；region/anchor 已编码的同一类别不重复交叉。

完整 factor one-hot、其他完整 categorical one-hot 及其完整外积同时存在时会精确线性相关，因此实际 interaction 不使用冗余的 full one-hot outer product，也不对任意 treatment-coded columns 逐列做 support mask。对每个 channel，完整 \(f_j\) 仍是 bias-free base projection 中唯一的完整 categorical baseline；orientation、geometry kind、region、anchor、side、availability 和 distance-bin 等其他 exhaustive categorical fields 在 \(u^{event}_j\) 与 \(u_r^{route}\) 中使用冻结的 reference coding。每个 base categorical field 的 reference 固定为其 split-neutral raw vocabulary 中稳定 ID 最小的合法 level，`NA` 仍由既有 availability mask 表达而不得被选作 reference；具体 reference 与 column order 写入 `EventFeatureManifest`。这里的 base reference 只定义主效应坐标，不定义 interaction、comparator 或科学 baseline。interaction factor vocabulary 只包含具名 factor-specific DNA events 或 RNA/RBP events；`OPEN_ONLY` 没有 factor identity，其 \(u_r^{int}\) 固定为零，只走 base block。context fields 之间只构造 factor×单个 coarse field 的二阶 interaction，不构造多个 fields 的联合笛卡尔积。

\[
u_r^{base}
=
\operatorname{concat}
\left[
u^{event}_{j(r)},\,
u^{route}_r
\right].
\]

对 channel \(M\) 的每个 context field \(t\)，令 \(\mathcal F_M\) 为按稳定 ID 排序的 raw factor/entity vocabulary，\(\mathcal L_{M,t}\) 为 split-neutral raw context-level vocabulary，raw cell universe 为 \(\mathcal C_{M,t}=\mathcal F_M\times\mathcal L_{M,t}\)。全部无约束 interaction 的最大秩固定为

\[
p^{max}_{M,t}
=
(|\mathcal F_M|-1)(|\mathcal L_{M,t}|-1).
\]

实际 support-closed basis 只由下面的 train raw-cell support 构造。对每个四角均通过支持门的 rectangle \(R=(f,f^*,l_a,l_b)\)，其中 factor pair 与 context pair 都按稳定 ID 规范排序，定义 raw-cell contrast vector \(d_R\in\{-1,0,1\}^{|\mathcal C_{M,t}|}\)：

\[
d_R(f,l_a)=d_R(f^*,l_b)=+1,
\qquad
d_R(f,l_b)=d_R(f^*,l_a)=-1,
\]

其余 raw cells 为零。按 `(factor pair, context pair)` 的稳定字典序遍历全部 supported rectangles，以 exact-rank greedy selection（或产生完全相同 canonical pivot set 的 exact RREF）保留能增加 span rank 的 \(d_R\)，得到 reference-independent matrix \(H^{support}_{M,t}\in\mathbb Z^{|\mathcal C_{M,t}|\times p^{support}_{M,t}}\)：raw cells 按固定 \(\mathcal C_{M,t}\) 顺序占 rows，canonical pivot contrasts 占 columns。route \(r\) 在该 field 的 interaction feature 等于其 raw cell \((f(r),l_t(r))\) 对应的 matrix row；每个 field 先在自身固定 segment 内补零到 \(p^{max}_{M,t}\)，再依 manifest 顺序 concat，因此总宽度恒为 \(p^{max}_M=\sum_t p^{max}_{M,t}\)：

\[
u_r^{int}
=
\operatorname{concat}_t
\operatorname{pad}_{p^{max}_{M,t}}
\left(
H^{support}_{M,t}[(f(r),l_t(r)),:]
\right).
\]

`m_M^{int}` 只标记 canonical supported-basis/rank-audit 后的 active padded positions；它不再是 treatment-column support mask。三个 runtime conditions 使用相同 \(u_r^{int}\) 宽度、column order 和 mask。该 basis 仍只进入现有 bias-free \(W_M^{int}\)，不增加 trainable module 或参数宽度；单个 basis coefficient不是科学 estimand。

\(u_r^{int}\) 是固定、稀疏、可审计的 factor-conditioned context-deviation basis，用于表达当前问题所需的 factor-specific positional grammar。V2 第一版不加入 factor×raw-coordinate、factor×motif-ID、全 event-feature×route-feature 笛卡尔积、三阶交叉或 score×factor×position。除非后续另行证明 motif score 是可比较的定量结合强度并修订 claim，第一版不声称学习了 motif-strength-dependent positional effect。`motif_id` 与 `motif_equivalence_family_id` 保留为 provenance，但不设高维 motif-ID embedding；V2 也不另设 per-factor network 或 nonlinear EventEncoder。accessibility-only event 的 motif score为零并使用明确的 `NA` orientation。这样同一个物理 event 可以拥有不同的 route-specific geometry 和 factor-conditioned positional deviation，但 attribution 和 neutralization 仍以 event \(j\) 为单位；贡献只由 §12 的 gauge-invariant output counterfactual 定义。

模型训练前，必须对完整固定 route-feature design \([u^{base},u^{int}]\) 执行 zero-column、exact-duplicate 与 rank audit。若 base block 非满秩，必须先修正或删除语义重复的 base coding并更新 manifest；不得通过牺牲 interaction 掩盖 base 冗余。base 满秩后，按上述 canonical interaction-column 顺序只保留能增加完整 design rank 的 columns；对每个 field 记录 retained column indices \(S^{active}_{M,t}\)，并在 raw-cell coordinates 中定义 \(H^{active}_{M,t}=H^{support}_{M,t}[:,S^{active}_{M,t}]\)，各 field 的 indices 共同定义最终 \(m_M^{int}\)。不得依靠 weight decay 在等价参数分解中任意选择一个解。raw support、candidate rectangles 与 comparator universe 不读取也不依赖 base reference；production rank audit 和 learned outputs 则只对 manifest 中冻结的 base coding 定义。

`EventFeatureManifest` 必须在实现前冻结并由三个 runtime conditions 共用，记录 factor/entity raw vocabulary与顺序、`OPEN_ONLY` 类别、\(u^{event}\)/\(u^{route}\)字段与 availability masks、每个 \(q_r\) field 的 raw level vocabulary/distance-bin boundaries、预声明 scientific context-pair universe、motif-score是否进入模型及其 calibration identity、orientation interaction policy，以及 \(p^{max}_{M,t}\) 与 potential schema。该 split-neutral vocabulary只能依据 catalog、reference和 train-independent calibration构建，不能读取 path outcome；train-only raw-cell support、supported rectangles、canonical pivot rectangles、combined rank audit、final padded-column order和 mask另存于 `InteractionSupportManifest`，未修订合同时不得追加匿名列。raw rectangle/basis/span 的 rank 使用整数 exact arithmetic；含连续 base features 的 full route-design audit 则冻结 column scaling、deterministic rank-revealing QR/SVD 实现及纯数值 tolerance，并保存 singular values/pivots，不能在看见 validation/test effect 后改变。

interaction support 先在尚未构造 interaction basis 的 raw biological cells \(b=(M,t,f,l)\) 上计算，其中 \(M\) 是 channel、\(t\) 是 context field、\(f\) 是 factor/entity raw level、\(l\in\mathcal L_{M,t}\) 是该 field 的 raw level。该门在 final route cap、gate admission 和 active-event filtering 之后计算，并对每个 \(b\) 至少保存：激活该 cell 的 distinct physical-event 数 \(N_b^{event}\)、distinct target-gene 数 \(N_b^{gene}\)、distinct active gate-key 数 \(N_b^{gate}\)，以及

\[
M_b^{inf}
=
\sum_{(i,g)\in train}
w_{i,g}\,
\mathbb 1\!\left[
\exists r:\ (M(r),t,f(r),l_t(r))=b,\ g(r)=g,\ \omega_{i,g,\tau(j(r))}>0
\right],
\]

即对每个合法 \((gene,cell)\) 最多计一次的 likelihood-informative molecule mass。四个下限由 train coverage audit 冻结；同一 channel 的所有 biological cells 使用相同阈值，validation/test 不得改变 support status，也不得逐 factor 调参。

同一 manifest 必须按 `(channel, context field)` 分别保存 `N_raw_rectangles_potential`、`N_four_corner_supported`、`N_support_span`、`N_rank_retained` 与固定 `N_padded=p^{max}_{M,t}`，并联结每个 raw biological cell的 distinct events、target genes、active gate keys 与 \(M_b^{inf}\)、每个 rectangle 的四角支持和逐 basis-column关闭原因。`OPEN_ONLY`不进入任何 factor-specific分母。若没有 supported rectangle，`N_support_span=0`并标记`not_applicable_no_supported_rectangle`；否则按 active span相对 support span标记`zero | partial | full`。rectangle count、support-span rank与combined-design rank不得混成一个 coverage数字。

科学 claim 另由 reference-independent 的 `RawInteractionContrastTable` 决定。对每个预声明 raw contrast \(q=(M,t,f,l_a:l_b)\)，其中 \((l_a,l_b)\) 是 field \(t\) 的 scientific context pair 而非 coding reference，先检查 focal cells \((M,t,f,l_a)\) 与 \((M,t,f,l_b)\)；再在同一 channel/context field 的其他具名 factors/entities 中寻找 comparator \(f^*\)，要求 \((M,t,f^*,l_a)\) 与 \((M,t,f^*,l_b)\) 也通过同一门槛。`OPEN_ONLY` 不得作为 comparator。令 \(\eta_{M,t}(f,l)\) 表示定义 contrast direction 所用的任意标量 raw-cell surface；它不是单个 learned coefficient 或 attribution estimand。每个合格 comparator定义 raw difference-in-differences：

\[
\delta_q(f^*)
=
\left[\eta_{M,t}(f,l_a)-\eta_{M,t}(f,l_b)\right]
-
\left[\eta_{M,t}(f^*,l_a)-\eta_{M,t}(f^*,l_b)\right].
\]

raw support 状态固定为：focal 任一 arm 不足时 `unsupported_focal_arms`；focal 两 arms 完整但没有任何完整 comparator 时 `within_factor_only`；存在至少一个完整 comparator 时先记 `four_corner_covered`。每个 q 另有一条 summary row；无 comparator 时其 `comparator_id=NULL`，以承载 q-level 的 `unsupported_focal_arms`、`within_factor_only` 或 not-applicable 状态。`within_factor_only` 只允许报告该 factor 在模型中的 context contrast，不允许称为 factor-specific positional grammar。

四角 coverage 仍不自动证明最终 active interaction design 能表达该 raw contrast。对每个 \((q,f^*)\)，使用与 \(H^{active}_{M,t}\) 相同的 raw-cell 顺序构造其 \((+1,-1,-1,+1)\) vector \(d_{q,f^*}\)，并以 exact arithmetic 检查

\[
\operatorname{rank}(H^{active}_{M,t})
=
\operatorname{rank}([H^{active}_{M,t},d_{q,f^*}]).
\]

除 q-summary 外，表中每个 \((q,f^*)\) 一行，保存 `contrast_in_active_span`、两侧 rank 和 comparator ID。还需把 \(d_{q,f^*}\) 按 field \(t\) 的 raw-cell lookup 提升到共同 route rows 得到 \(\widetilde d_{q,f^*}\)，并令 \(Z_{-t}^{support}\) 包含 frozen base block 和其余 context fields 在 combined pruning 前的全部 supported interaction columns。若在同一冻结 full-design rank tolerance 下

\[
\operatorname{rank}(Z_{-t}^{support})
=
\operatorname{rank}([Z_{-t}^{support},\widetilde d_{q,f^*}]),
\]

则该 raw direction 也可由其他 field semantics 表示，必须标记 `cross_field_context_not_separable` 并列出 aliased field IDs；它只能进入 joint-field diagnostic，不能获得 field-specific grammar admission。每个 comparator 独立保存 `contrast_in_active_span` 与 `cross_field_context_separable`，后者不以前者为前提。q-level design-admission 只汇总两者均为 true 的 comparator IDs：至少一项通过才得到 `factor_specific_grammar_estimable`；否则只要任一完整 comparator 被 \(Z_{-t}^{support}\) 完全表示，就优先标记 `cross_field_context_not_separable`，使 combined pruning 中先、后出现的两个 aliased fields 得到相同降级；仅在不存在 cross-field alias 且完整 comparators 全部不在 active span 时才为 `raw_contrast_not_in_active_span`；其余分别为 `within_factor_only` 或 `unsupported_focal_arms`。`factor_specific_grammar_estimable` 只证明该预声明 raw DID 在 active design 中可表达且未被其他 field 完全替代，不证明训练后效应非零、有方向或采用该机制。当前 V2 的数值 estimand 仍是 §12 的完整 event/set output counterfactual，不把 basis coefficient 或未另行定义的 raw-cell \(\eta\) 当作 learned grammar effect；若要声称“factor \(f\) 的位置偏好相对 \(f^*\) 非零”，需未来另行冻结 comparator-linked gauge-invariant output DID，本版不作该更强 claim。一个 comparator 只支持相对于该 comparator 的 design contrast，不能推广成相对于所有 factors 的特殊性；正式 design-scope 表述必须列出实际通过两道 gate 的 comparator IDs。

raw biological-cell support、canonical supported-rectangle span、raw contrast status、comparator IDs、contrast-in-span 与 cross-field separability 共同构成 scientific claim gate；不存在由任意 factor/context reference 决定的 interaction column。对只改变 base categorical reference 的离线重编码审计，raw support/contrast tables 必须相同，并以 exact ranks 验证对应 \(H^{active}\) spans 相同；否则 manifest validation 失败。这只证明 design-admission invariance：AdamW 与训练动力学只对 production manifest 中冻结的 stable-ID base reference 定义，不声称 alternative non-orthogonal base recodings 会得到逐值相同的 learned outputs。未获 factor-specific design admission 的 events 仍保留 physical event、gate、factor baseline、共享 route-context baseline 以及设计允许的完整模型 prediction；这里限制的是 positional-grammar 表述，不声称后续非线性网络不存在隐式 dependence。

#### Model-injection equivalence

在 final cap、gate-key admission、canonical interaction basis、rank closure 和最终 active-basis mask \(m_M^{int}\) 全部冻结后，必须只对 `model_active=true` events 构建 reporting-only 的 `ModelInjectionEquivalenceIndex`。它只读取冻结的 catalog/design tensors，不读取 path outcome、validation/test support、learned \(W\)、checkpoint 输出或 attribution magnitude，并由三个 runtime conditions 共用。

对 modality \(M\) 的 event \(j\)，定义最终 per-edge fixed-feature aggregates：

\[
\beta^M_{j,e}
=
\sum_{r:j(r)=j}
R^M_{r,e}u_r^{base},
\qquad
\iota^M_{j,e}
=
\sum_{r:j(r)=j}
R^M_{r,e}
\left(m_M^{int}\odot u_r^{int}\right).
\]

其 architecture-level exact injection signature 固定为：

\[
\sigma(j)
=
\left(
g(j),\,M(j),\,\tau(j),\,
\left[e,\beta^M_{j,e},\iota^M_{j,e}\right]_{e\text{ 按稳定 edge ID 排序}}
\right),
\]

并定义 \(j\equiv_{inj}k\iff\sigma(j)=\sigma(k)\)。signature 对该 gene 的完整稳定 edge axis 物化；没有注入的 edges 使用 exact-zero \((\beta,\iota)\)，不得因 sparse omission rule 不同而拆组。聚合按稳定 route ID 顺序执行，`-0.0` 随后 canonicalize 为 `0.0`。等价判断只精确比较 signature 中的 gene、modality、gate key 和这些最终 per-edge aggregates；individual route IDs、单条 route weights 及其分解只作 provenance，分解不同但 aggregate 完全相同仍属于同组。categorical/reference codes、availability masks、连续值和 active-basis interaction values 已经通过 \((\beta,\iota)\) 进入比较；任一最终 aggregate 精确不同才拆组。不得使用 tolerance、rounding、近邻或 fuzzy hash 建组。不同 gate keys 在有限 train cells 上碰巧相同、当前 cell 的 \(G=0\)，以及训练后由 \(W_M\) 或 \(W_X\) null space 造成的 checkpoint-specific collision，都不属于该冻结 index。

每个 index row 至少保存：

```text
model_injection_group_id
target_gene_id
modality
gate_key_id
member_event_ids and member_count
ordered edge_ids
per-edge beta_base and iota_masked_interaction
member_route_ids and anchor_region_ids
footprint_relation = identical_interval | overlapping_under_frozen_rule | mixed_or_disjoint
motif_family_relation = same_motif_equivalence_family | distinct_non_equivalent_families | accessibility_only | mixed_or_NA
physical_collapse_status = correctly_distinct | should_have_collapsed_error
attribution_policy = singleton_event_primary | exact_injection_set_primary
```

相同 signature 只证明这些 events 对任意 cell 和任意合法 shared projection 参数具有相同的单份 pre-LayerNorm 注入，不证明它们是同一物理证据。该 index 不改变 `PhysicalEventTable`、`EventRouteTable`、cap、route weights、event burden 或模型 tensors；不选择 representative event，不把其他 members 改写成 `source_motif_ids`，不删除一份注入，也不乘除 group size。每个 physical event 仍按 §8.1 独立进入求和。只有已经满足 §5.1 biological physical-collapse key 与冻结 overlap/equivalent-PWM rule 的重复 rows 才应在 routing 前 collapse；若这类 rows 仍出现在一个 injection group 中，标记 `should_have_collapsed_error` 并令 catalog validation 失败，而不能由该 index 静默补救。相同或重叠 footprint 上的非等价 PWM/binding-mode annotations 即使 signature 相同也保留各自 physical rows 与 forward terms，但不得作单 motif、单 binding mode 或唯一位点归因。

对 multi-member equivalence class \(E\)，§12 的主要 attribution selector 必须使用其确定性 route union \(\mathcal R(E)=\bigcup_{j\in E}\mathcal R(j)\) 并重新 forward。单 member neutralization 可以保留为 `exchangeable_member_counterfactual` 诊断，但不得进入具名 event/motif/位点机制 summary；joint-set effect 也不得由 `member_count × single-member effect` 代替。任一 event、factor 或 anchor selector 若只覆盖该 class 的真子集，必须标记 `partial_model_injection_group` 并降级为诊断。这个 derived set 不是新训练头或第四种 primitive learnable selector。

### 5.6 Self-factor policy

若 `target_gene_id` 属于 event 的 `activity_gene_ids`，主模型保留该 event，并记录 `is_self_factor=true`；DNA 和 RNA 使用同一规则。理由是短读长总表达并不是长读长相对 path label，而且 RBP/TF autoregulation 可能是真实机制。任何 target-path count、DTU label 或 long-read-derived usage 都不得进入 activity proxy。self 与 non-self events 必须分层报告，self-factor attribution 只能称为 conditional association，不能仅凭该输入宣称“自身表达导致自身剪接”。

## 6. Cell-specific gates

### 6.1 Source scale and raw signals

RNA factor activity 与 ATAC peak accessibility 都在源端使用固定 CP10K library normalization 后的 `log1p` 非负值；target sum 固定为 10,000。令 \(x_{i,f}\) 为 factor gene \(f\) 的短读长原始 count，\(\mathcal G^{RNA}\) 为冻结的完整短读长 library gene axis，\(L_i=\sum_{g'\in\mathcal G^{RNA}}x_{i,g'}\) 为该 RNA cell 的 library denominator；分母不得只对 activity-factor 子集或当前 event catalog 求和。unique factor 与 `factor_equivalence_group` 的 activity 分别固定为：

\[
F_{i,f}
=
\log\!\left(1+10^4\frac{x_{i,f}}{L_i}\right),
\qquad
F_{i,h}
=
\log\!\left(1+10^4\frac{\sum_{f\in\mathcal F_h}x_{i,f}}{L_i}\right).
\]

group activity 必须先在原始 count 空间对冻结的 `activity_gene_ids` 求和，再做一次 CP10K-log1p；不得平均成员的 log-normalized expression，也不得按当前细胞选择最高表达成员。ATAC 必须先对每个真实 ATAC cell 归一化，再按冻结的 RNA-to-ATAC neighbor weights 映射到 RNA cell。V2 不对 factor activity 与 accessibility 分别 z-score 后再相乘，也不在乘积之后再做第二次 `log1p`。

短读长 source gene axis 在 catalog 构建前冻结，且上述完整 \(\mathcal G^{RNA}\) 同时定义 CP10K denominator。一个 activity entity 只有在其全部 `activity_gene_ids` 都各自唯一存在于该 axis 时才可进入 model-active catalog；任一成员缺轴或 ID mapping 不唯一时，整个 entity 标记 `invalid_activity_axis`，不得对剩余成员静默求和。对 gene axis 合法且 \(L_i>0\) 的 RNA cell，稀疏矩阵中的零是 observed zero，并令 \(m^F_{i,h}=1\)；只有整项 RNA observation 缺失、QC failure 或 \(L_i=0\) 时令该 cell 的 \(m^F_{i,h}=0\)。同一 activity entity 的 mask 对其全部 genes/events 共享。

令 \(F_{i,h}\ge 0\) 为 activity entity \(h\) 的 factor/group proxy，\(A_{i,p}\ge 0\) 为 mapped peak accessibility。三个 raw signals 定义为：

\[
b^{RNA}_{i,h}=F_{i,h},
\]

\[
b^{DNA}_{i,h,p}=F_{i,h}A_{i,p},
\]

\[
b^{Open}_{i,p}=A_{i,p}.
\]

DNA raw signal 必须先在非负空间形成完整乘积，之后才允许中心化和尺度控制。

ATAC neighbor-weight concentration 只作为每个 RNA cell 的映射支持度诊断，不进入模型 gate。令 \(\alpha_{i,\ell}\) 为该 RNA cell 的有效、和为一的 ATAC-neighbor weights，\(K\) 为预声明的目标 neighbor 数，并强制 \(0<K_i\le K\)。当 \(K_i>0\) 时记录：

\[
\operatorname{ESS}^{ATAC}_i
=
\frac{1}{\sum_{\ell=1}^{K_i}\alpha_{i,\ell}^2},
\qquad
\operatorname{Even}^{ATAC}_i
=
\frac{\operatorname{ESS}^{ATAC}_i}{K_i},
\qquad
\operatorname{Cover}^{ATAC}_i
=
\frac{K_i}{K}.
\]

其中 \(\operatorname{Even}^{ATAC}_i\) 只描述现有 neighbor weights 的均匀程度，\(\operatorname{Cover}^{ATAC}_i\) 只描述目标 neighbor 数的覆盖比例；旧定义 \(\operatorname{ESS}_i/K=\operatorname{Even}^{ATAC}_i\operatorname{Cover}^{ATAC}_i\) 把二者混在一起。它们都不等于 mapping correctness、peak-specific measurement precision 或 biological reliability。一个高度匹配的单 neighbor 与同一 neighbor 的多个重复副本可以产生相同 mapped accessibility，却有不同 ESS；多个同样遥远的均匀 neighbors 也可以有最大 ESS。因此 ESS、evenness、coverage、最大 neighbor weight、最近/加权平均距离及 neighborhood-consistency status 只保存到 mapping/support metadata，用于 QC 审计和分层报告；它们不得进入 gate tensor、\(\mu/\sigma\) 拟合权重、event aggregation、path logits、loss 或 attribution multiplier。若没有合法 ATAC neighbor，则固定 \(m_i^A=0\)，上述诊断记为 `not_estimable`，不得计算空分母或把该细胞当作 observed zero。\(A_{i,p}\) 才是按冻结 weights 映射的 peak-specific accessibility，静态 `peak_support` 才是 peak-specific catalog 支持。

映射资格由独立的冻结布尔量 `mapping_valid` 决定，并体现在 \(m_i^A\) 中。其数据合同至少要求：RNA/ATAC cells 通过各自 QC，stage/developmental-system 与 donor constraints 符合预声明的 pairing policy，neighbor distances 为有限值且位于 train-only 冻结的 admissible range，至少一个合法 neighbor，weights 非负且归一化，并执行预声明的 neighborhood-consistency audit。`mapping_valid` 必须依据绝对匹配/QC 条件；局部 atlas density、\(K_i\)、ESS、evenness、coverage 或 weight concentration 不得单独或与其他条件联合参与该布尔判定。当 \(K_i=1\) 时 neighbor agreement 记为 `not_estimable`，不能仅因此把一个满足其他绝对匹配条件的单 neighbor 判为无效。任一必需 QC 条件失败时 \(m_i^A=0\)。具体 embedding、distance metric、absolute distance threshold、可估计时的 consistency policy、donor policy 与 \(K\) 必须在 mapping manifest 中于 outcome evaluation 前冻结。除非未来有 paired truth 或显式 peak-specific measurement-error model 校准连续 uncertainty，否则 V2 不对有效 mapped accessibility 做额外乘性收缩。

### 6.2 Train-only centering and scale

多个 motif hits 可以共享同一个动态 gate key，避免为相同生物学信号重复估计 baseline。用 \(\tau\) 专指 gate-key index，避免与 §3/§10 的 EC row \(k\) 混淆。RNA key 固定为 \(\tau=(g,h)\)，factor-specific DNA key 固定为 \(\tau=(g,h,p)\)，accessibility-only key 固定为 \(\tau=(g,p)\)，其中 \(h\) 是 `activity_entity_id`；每个 event \(j\) 只引用一个 \(\tau(j)\)，并在 `PhysicalEventTable.gate_key_id` 保存其稳定 ID。

令 \(w_{i,g}=\sum_{k\in\mathcal K^{inf}_{i,g}}n_k\) 为 train split 中该 cell-gene 的 likelihood-informative molecule mass，\(m^F_{i,h}\) 为 activity-entity observation mask，\(m^A_i\) 为 mapped ATAC observation mask。不同 channel 的有效权重为：

\[
\omega^{RNA}_{i,g,\tau}
=w_{i,g}m^F_{i,h},
\]

\[
\omega^{DNA}_{i,g,\tau}
=w_{i,g}m^F_{i,h}m^A_i,
\]

\[
\omega^{Open}_{i,g,\tau}
=w_{i,g}m^A_i.
\]

对每个 gate key \(\tau\)，只用 train cells 拟合并冻结：

\[
\mu^{train}_{g,\tau}
=
\frac{\sum_i\omega_{i,g,\tau}b_{i,\tau}}
{\sum_i\omega_{i,g,\tau}},
\qquad
\left(\sigma^{train}_{g,\tau}\right)^2
=
\frac{\sum_i\omega_{i,g,\tau}
\left(b_{i,\tau}-\mu^{train}_{g,\tau}\right)^2}
{\sum_i\omega_{i,g,\tau}}.
\]

唯一尺度控制是 raw signal 形成之后的 train-only、molecule-weighted z-score。validation/test 只应用冻结的 \(\mu^{train}\) 与 \(\sigma^{train}\)：

\[
G^{RNA}_{i,g,\tau}
=
m^F_{i,h}
\frac{b^{RNA}_{i,h}-\mu^{train}_{g,\tau}}
{\sigma^{train}_{g,\tau}},
\]

\[
G^{DNA}_{i,g,\tau}
=
m^F_{i,h}m^A_i
\frac{b^{DNA}_{i,h,p}-\mu^{train}_{g,\tau}}
{\sigma^{train}_{g,\tau}},
\]

\[
G^{Open}_{i,g,\tau}
=
m^A_i
\frac{b^{Open}_{i,p}-\mu^{train}_{g,\tau}}
{\sigma^{train}_{g,\tau}}.
\]

上式是“train-only molecule-weighted standardized raw residual × observation masks”。固定 mapped \(A_{i,p}\) 与 \(m_i^A\) 时，DNA/Open gate 不再受额外的 atlas-density 或 neighbor-weight-concentration 乘性因子影响；neighbor set/weights 若确实改变 mapped \(A_{i,p}\)，则仍可通过观测 accessibility 改变 gate。映射能否进入模型只由 \(m_i^A\) 表示，neighbor-support diagnostics 仅用于 QC 与分层报告。这样不会把 atlas sampling density 本身当成生物学调控强度，也不会用未校准的 ESS 近似 measurement-error shrinkage。

gate 本身不增加 clipping、learned temperature、按 event count 的除法或第二套 gate 尺度变换。§8.1 的固定 joint-input pre-normalization 发生在 event aggregation 与联合输入投影之后，只控制进入 GraphGPS 的 token 数值尺度；它不是 gate 变换，也不改变本节定义的 \(G\)。gate-key admission 不能只检查分母非零：实现前必须在不读取 path outcome 的 train gate audit 上预声明并冻结四个下限，即有效 unique cell 数 \(n^{valid}_{\tau}=\sum_i\mathbb 1[\omega_{i,g,\tau}>0]\)、gate-level weighted effective cell count \(n^{eff,gate}_{\tau}=(\sum_i\omega_{i,g,\tau})^2/\sum_i\omega_{i,g,\tau}^2\)、likelihood-informative molecule mass \(M^{inf}_{\tau}=\sum_i w_{i,g}\mathbb 1[\omega_{i,g,\tau}>0]\)，以及 channel-specific weighted raw-signal standard-deviation floor \(\sigma_{min,channel}\)。\(n^{eff,gate}_{\tau}\) 只衡量一个 gate key 的训练支持，不是 §6.1 的 ATAC-neighbor ESS，也不连续乘入任何 cell 的 gate。具体数值阈值由 train-only coverage audit 写入 `GateAdmissionManifest` 后、在任何 validation/test model comparison 或正式训练前冻结；同一 channel 的所有 gates 使用同一组阈值，不能逐 factor/gene 调参。只有前三项通过且 \(\sigma^{train}_{g,\tau}>\max(\sigma_{min,channel},10^{-8})\) 时 `gate_key_active=true`。失败原因按 `no_train_observation`、`insufficient_valid_cells`、`insufficient_gate_effective_cells`、`insufficient_informative_molecules`、`insufficient_train_variation` 的固定顺序保存为可多值列表；不得使用 epsilon fallback 强行除法。资格只由 train split 决定并冻结。event-level `model_active` 再严格按 §5.1 的 conjunction 得到；未激活 events 继续留在完整 MotifEventCatalog 做覆盖率审计，但从训练、event aggregation 和 attribution tensors 中排除，而不是保留为静默零列。

每个 active gate key 还必须在与 \(\mu/\sigma\) 完全相同的 train cells 和 \(\omega_{i,g,\tau}\) population 上冻结 raw signal 的 support summary：observed minimum/maximum 和预声明的 lower/upper weighted quantiles。validation/test 以及 source-proxy/member-count perturbation 都继续用冻结 \(\mu/\sigma\) forward，既不 clipping 也不重估尺度；但若 raw value 越过 train min/max，则记录 `out_of_train_range=true`，若越过冻结 quantile interval 则记录 `out_of_train_quantile_support=true`。记录值必须区分 raw signal \(b_{i,\tau}\)、标准化残差 \(z_{i,g,\tau}=(b_{i,\tau}-\mu^{train}_{g,\tau})/\sigma^{train}_{g,\tau}\) 和乘 observation masks 后真正进入模型的 final gate \(G_{i,g,\tau}\)，不得把三者都称为 `standardized_gate_value`。超支持输入可用于明确标记的外推预测，但其 event-level mechanism attribution 不进入 primary supported-claim set，只能作为 `model_extrapolation` 分层报告。

真实观测零值与缺失必须区分。`observed=1, value=0` 是测得的零值，经标准化后可以成为相对训练基线的负信号；`observed=0` 表示没有有效观测，对应 cell-event gate 严格为零，但不会把该 event 从其他有观测细胞中删除。

### 6.3 Gate evidence-separation audit

单个 gate 的变异与支持充分，不表示多个动态证据来源已在训练队列中被独立分开。对同一 gene 内两个 active gate keys \(\tau,\tau'\)，只在二者均有有效观测且 \(w_{i,g}>0\) 的 train cells

\[
\mathcal I_{g,\tau,\tau'}
=
\{i:m_{i,\tau}=m_{i,\tau'}=1,\ w_{i,g}>0\}
\]

上计算 molecule-weighted Pearson correlation

\[
r^w_{g,\tau,\tau'}
=
\frac{
\sum_{i\in\mathcal I}w_{i,g}(G_{i,g,\tau}-\bar G^w_{g,\tau})(G_{i,g,\tau'}-\bar G^w_{g,\tau'})
}{
\sqrt{
\sum_{i\in\mathcal I}w_{i,g}(G_{i,g,\tau}-\bar G^w_{g,\tau})^2
\sum_{i\in\mathcal I}w_{i,g}(G_{i,g,\tau'}-\bar G^w_{g,\tau'})^2
}
},
\]

其中加权均值使用同一 \(\mathcal I\) 与 \(w_{i,g}\)，并记录联合有效样本量

\[
n^{eff,col}_{g,\tau,\tau'}
=
\frac{\left(\sum_{i\in\mathcal I}w_{i,g}\right)^2}
{\sum_{i\in\mathcal I}w_{i,g}^2}.
\]

其中 \(m_{i,\tau}\) 按 gate channel 分别等于 \(m^F_{i,h}\)、\(m^F_{i,h}m^A_i\) 或 \(m^A_i\)，不另建 observation mask。`GateCollinearityAudit` 必须覆盖同一 factor/entity 的 DNA–RNA gates，以及不同 factors/entities 的同通道和跨通道 gates；最小联合支持 \(n^{col}_{min}\) 与绝对相关阈值 \(\rho_{col}\) 在 validation/test attribution 和 full-cohort training 前由 train-only、outcome-blind gate audit 冻结。联合支持不足标为 `evidence_separation_not_estimable`；\(|r^w|\ge\rho_{col}\) 标为 `correlated_evidence`；其余只能称 `no_high_pairwise_collinearity_detected`，不能称为 identifiable。完全相同或相反的 finite-variance gate vectors 必须保留为 \(|r^w|=1\)；任一方在共同 cells 中方差为零时标为 `evidence_separation_not_estimable`，不得用 epsilon 制造相关系数。

超过阈值的 pair 构成 reporting-only 无向图，其 connected components 定义 `correlated_evidence_set` 以便联合 neutralization；原始 pairwise edges 与 correlations 必须保留，因为同一 component 不表示所有成员两两高相关。该 set 只说明当前训练细胞缺少把这些 evidence sources 独立分开的变化，与 motif identity 不可区分的 `factor_equivalence_group` 不同；不得据此合并 event/factor identity、改变 gate、删除单事件 attribution 或重训模型。属于该 set 或 separation not estimable 的单来源 \(\Delta\rho\) 仍可作为带标记的模型反事实报告，但不得声称“唯一由 DNA 而非 RNA”或“唯一由 factor A 而非 factor B”产生；优先补充整组联合 neutralization。任意数量 seeds 的一致性和外部扰动都不能改写该 checkpoint 的 train-only evidence-separation status。具有 target engagement、matched control 且确认 correlated partners 未被共同改变的来源特异扰动，只能支持一个独立、明确限域的外部机制一致性或因果 claim；普通 factor KD 若同时改变 DNA/RNA 两路，不能解除 DNA-vs-RNA 降级。只有把产生正交 variation 的新数据纳入后续模型并重新冻结、训练和审计，才可能改变新版 status。

### 6.4 Gate and intervention semantics

gate 为零表示该动态证据处于训练基线或没有有效观测，不表示 factor 生物学缺失，也不表示基因组 motif 被删除。由于多个 motif events 可以共享同一个 gate key，single-event evidence neutralization 只能将该 event 的全部 routed terms 置零；这些 terms 按 §8.1 定义为 \(R_{r,e}G_{i,g,\tau(j(r))}(W^{base}u_r^{base}+W^{int}(m^{int}\odot u_r^{int}))\)。共享的 \(G_{i,g,\tau(j)}\) 本身保持不变。gate-key baseline neutralization 才把某个 \(G_{i,g,\tau}\) 置零，并同时影响所有引用该 key 的 events；factor/group-level baseline neutralization 则删除该 factor/group 对应的全部 event routes。`source_proxy_perturbation` 直接修改已经按 §6.1 定义的 \(F_{i,h}\) 或 mapped \(A_{i,p}\)，再用冻结的 \(\mu^{train}/\sigma^{train}\) 重算受影响 gates；`member_count_perturbation` 修改原始 \(x_{i,f}\)，并重算所有包含该成员的 unique/group proxies 与 gates。独立真实 perturbation library 则按其完整 counts 和 library denominator 重建新的 `observed_library_context`。single-event、gate-key、factor/group、source-proxy、member-count 与 observed-library 操作必须按各自 scope 分开命名和报告，不能统称为 raw perturbation。

## 7. Event-to-token routing

physical event 与 processing-edge token 之间通过 EventRouteTable 和冻结的稀疏 route-edge matrix \(R_{r,e}\) 连接，其中 \(j(r)\) 表示 route \(r\) 所属 event。每条 route 只指向一个 edge，矩阵值等于 `route_weight`，并要求每个 event 的全部保留 routes 总权重为一：

\[
\sum_{r:j(r)=j}\sum_eR_{r,e}=1.
\]

`anchor_region_id` 由 graph identity 决定，而不是由 choice 决定：site region 使用 `gene_id:site:node_id`，edge region 使用 `gene_id:edge:edge_id`。TSS/donor/acceptor/PAS 分别使用预声明、transcript-oriented 的 modality-specific scan windows。RNA 第一版的候选区域只允许这些 site-centered windows，以及另行预声明最大长度内的短 exon interval；不得扫描完整长 intron 或把整条任意长度 processing edge 当作 RNA motif window。DNA edge region 可使用与真实 ATAC peak 相交的该 processing-edge 冻结 genomic interval，用于 intragenic accessibility/TF evidence；无 peak 支持的 factor-specific DNA event 不得由整条 edge 扫描产生。重叠窗口不合并，DNA/RNA 共享同一 structural anchor ID 但保留各自实际窗口和 modality。窗口越过 reference 或 gene contract 边界时只做显式 clipping 并记录原始/实际边界；短 exon 最大长度及所有 site-window flanks 在 motif/routing manifest 中冻结。

所有 event、anchor 和 edge intervals 统一使用 reference-build 上的 0-based、half-open genomic coordinates。physical event 的位置固定为 canonical interval center \(x=(start+end)/2\)，允许半整数；任何距离计算都不得在负链上改用未定义的整数取整。令 \(s_g=+1\) 表示正链、\(s_g=-1\) 表示负链。`site_window` route 先用严格谓词 `start < a < end` 判断 canonical interval 是否跨越 anchor boundary \(a\)；该条件优先，并编码 `OVERLAP_ANCHOR`。否则 `signed_distance_bp` 固定为 \(s_g(x-a)\)，负值/正值分别编码 `UPSTREAM`/`DOWNSTREAM`；模型中的连续距离再除以该 modality/anchor 的冻结 flank scale。其 edge-relative fields 为 `NA`，interval 仅与 boundary 相切（`end=a` 或 `start=a`）不算 overlap。

`edge_contained` 只允许正 span 的 half-open interval \([g_{lo},g_{hi})\)，\(L=g_{hi}-g_{lo}>0\)，并以 canonical center 满足 \(g_{lo}\le x<g_{hi}\) 作为唯一 membership predicate；不额外要求整个 motif/peak interval 被 edge 包含。从 transcript 5′ boundary 到 event center 的 oriented distance 定义为正链 \(d_{5'}=x-g_{lo}\)、负链 \(d_{5'}=g_{hi}-x\)，并令 \(d_{3'}=L-d_{5'}\)；`edge_relative_position=d_{5'}/L\in[0,1]`。该 route 不存在 site-centered signed distance，`signed_distance_bp=NA`，模型接收 `edge_relative_position` 及 \(\log(1+d_{5'})\)、\(\log(1+d_{3'})\)，anchor side 固定为 `WITHIN_EDGE`。两类 geometry 不得用零值代替不适用字段，必须附显式 availability/geometry-kind one-hot。

TSS/promoter event 路由到该 TSS 的 outgoing processing edges，PAS event 路由到该 PAS 的 incoming processing edges，位于允许的短 exon 或 DNA edge interval 内的 event 路由到对应 processing edge，splice-site window event 按 strand-aware endpoint role 路由。同一物理 hit 可以产生多个 route records；cap 后保留 routes 使用等权并在 event 内归一化，距离与 role 由 \(u_r^{route}\) 表达，不再通过距离重复加权。若没有合法 route，event 标记 `no_legal_route` 并退出 model-active catalog。绝对坐标只用于构建并审计 routing；模型只接收相对几何。不得在模型运行时通过最近距离或 fallback 链决定 anchor。

上述 sum-to-one 是冻结的 per-physical-event **partition convention**：它保持 event 的 route \(L_1\) mass，不是生物学定律，也不保证单 edge 注入、post-LayerNorm token state 或最终 output effect 对 route degree 不变。同一 anchor 的 \(d\) 条一对一等价 routes 在 pre-GraphGPS 每 edge 获得 \(1/d\) 权重，是当前方程的直接结果；后续 nonlinear effect 是否随 \(d\) 系统变化则由 `RouteDegreeCapAudit` 判断。该 audit 只审计当前 production \(R\)，不改变这里的方程，也不提供 broadcast、\(1/\sqrt d\) correction、learned/adaptive routing 或 site-carrier runtime switch。

一个物理 motif hit 即使拥有多个 routes 仍只有一个 event identity。single-event neutralization 必须一次删除其全部 routes，factor perturbation 必须一次更新该 factor/entity 的全部 events。anchor-region neutralization 只删除属于该 `anchor_region_id` 的 route terms；同一 event 在其他 anchor regions 的 routes 保留。routing 只是结构引用，不能因关联多个 edges 或 reporting choices 而复制 gate、复制 event identity 或制造多个伪 event。

## 8. Cell-conditioned gene graph encoder

### 8.1 Three-block input

令 \(c_e\) 为 edge \(e\) 的静态 CIS 特征，\(u_r^{base}\) 与 \(u_r^{int}\) 为 §5.5 定义的 route-aware fixed features，\(m^{int}\) 为冻结的 canonical interaction active-basis mask，\(j(r)\) 为其 physical event，\(\tau(j)\) 为该 event 引用的 active gate key。DNA 和 RNA 各使用一对可训练、bias-free、channel-shared 的线性投影 \((W_D^{base},W_D^{int})\) 与 \((W_R^{base},W_R^{int})\)：

\[
G^D_{i,g,r}
=
\begin{cases}
G^{DNA}_{i,g,\tau(j(r))}, & j(r)\text{ 是 factor-specific DNA event},\\
G^{Open}_{i,g,\tau(j(r))}, & j(r)\text{ 是 accessibility-only event}.
\end{cases}
\]

\[
d_{i,e}
=
\sum_{r\in\mathcal R^{DNA}}
R^{DNA}_{r,e}\,G^D_{i,g,r}
\left[
W_D^{base}u_r^{base}
+W_D^{int}\left(m_D^{int}\odot u_r^{int}\right)
\right],
\]

\[
r_{i,e}
=
\sum_{r\in\mathcal R^{RNA}}
R^{RNA}_{r,e}\,G^{RNA}_{i,g,\tau(j(r))}
\left[
W_R^{base}u_r^{base}
+W_R^{int}\left(m_R^{int}\odot u_r^{int}\right)
\right].
\]

模型输入为：

\[
x_{i,e}=\operatorname{concat}[c_e,d_{i,e},r_{i,e}].
\]

同一 token 内的 active events 在窄联合特征投影后直接求和。\(W_D^{base/int}\) 在全部 DNA 与 accessibility-only events 间共享，\(W_R^{base/int}\) 在全部 RNA events 间共享，二者不跨 channel 共用。固定的无冗余 \(u_r^{int}\) 保证有训练支持的不同 coarse context 类别下交换 factor–position 配对时，投影前 aggregate feature 可辨；同一 context bin 内的碱基级位置差异被有意视为等价。可训练投影仍可以学到相同或零映射，因此本合同不强迫最终 token state 或预测必须不同。拆分 base/int 投影只为实施 §15.5 的层级收缩，不增加非线性模块；与单个大矩阵相比，它只显式标出两组既有列的参数边界。

在给定模型参数与 route catalog 下，对任一 modality \(M\) 的单个 physical event \(j\)，定义其在 token \(e\) 上的投影方向

\[
v^M_{j,e}
=
\sum_{r\in\mathcal R^M:\,j(r)=j}
R^M_{r,e}
\left[
W_M^{base}u_r^{base}
+W_M^{int}\left(m_M^{int}\odot u_r^{int}\right)
\right].
\]

对固定观测 cell 的 single-event evidence neutralization，对应 modality aggregate 满足

\[
a^{M,full}_{i,e}-a^{M,(-j)}_{i,e}
=
G^{obs}_{i,g,\tau(j)}v^M_{j,e},
\qquad
a^{DNA}=d,\quad a^{RNA}=r.
\]

若合法的 gate-key perturbation 只把共享 key \(\tau\) 从 \(G_1\) 改为 \(G_2\)，则它必须同步改变全部引用该 key 的 events，并满足

\[
a^M_e(G_2)-a^M_e(G_1)
=
(G_2-G_1)
\sum_{j:\tau(j)=\tau}v^M_{j,e}.
\]

因此 event/gate-key routed terms 在 joint projection 和 pre-LayerNorm 之前对 **final gate** \(G\) 是严格仿射的；这只是代数注入关系，不允许把共享 gate 固定在其他 events 上、只对一个 event 做伪 source-dose sweep。\(G\) 是 §6 定义的 observation-mask-weighted standardized expression/accessibility proxy residual，不是 factor 浓度或 occupancy。V2 不增加 event attention、额外 factor embedding table、EventEncoder、factor/event-specific Hill function、learned threshold、mixture-of-experts、learned routing 或可切换 aggregation。

为保留多位点证据，aggregation 不除以 event count，也不乘 \(1/\sqrt{N}\)。但 per-anchor-group cap 不限制同一 token 从多个 groups 接收的总 event 数；如果高密度 events 共享 gate 且投影方向相近，未经控制的 token norm 会在进入 self-attention 前放大。V2 因此把联合输入先投影到 GraphGPS hidden width，再使用固定、无 affine 参数的 per-token pre-LayerNorm：

\[
y_{i,e}=W_Xx_{i,e}+b_X,
\qquad
\widehat y_{i,e}
=
\operatorname{LayerNorm}
\left(y_{i,e};\operatorname{elementwise\_affine}=\mathrm{false},
\epsilon=10^{-5}\right).
\]

局部与全局 GraphGPS 分支都只读取 \(\widehat y_{i,e}\)；已有 block-output normalization 继续保留。该 joint pre-normalization 不分别重标定 CIS、DNA 或 RNA，也不是按 event count 做平均：event sum 仍先改变动态 block 相对于 CIS 和其他 modality 的方向，随后才把进入 attention 的整体 token norm 限制在固定量级。对 centered direction 非退化且仅整体放大同一 aggregate direction 的情形，burden 对下游表示会趋于饱和；新增方向不同的 events 时不保证效应单调或饱和。无论哪种情况，都不能把 pre-normalization event sum 或最终预测解释为无界、线性可加的生物学剂量。event count、span、§5.1 的 \(B^{gate}\)、cap saturation、\(\lVert d_{i,e}\rVert_2\)、\(\lVert r_{i,e}\rVert_2\) 和 pre/post-normalization token norms 只进入诊断与分层报告，不作为额外模型特征或样本权重。

### 8.2 One-layer GraphGPS

三块输入经上述 joint projection 与固定 pre-normalization 后，共同经过一个浅层 gene-level GraphGPS：

\[
H_{i,g}=\operatorname{GraphGPS}(G_g,\widehat Y_{i,g}).
\]

GraphGPS 恰好保留一个 block：局部分支沿 processing-edge line graph 做 message passing，全局分支在同一 gene-cell instance 内做 self-attention。全局分支使一层具有全基因感受野；这只表示任意 token 的信息可以在一次 attention 中相互到达，不等于模型能够表达或从数据中辨识任意高阶 choice dependency。共享 path MLP 表达的是隐式 path-context dependence，不是具名、可单独估计的 choice-pair potential。V2 不堆叠多层 GraphGPS，也不做 encoder architecture search。

joint pre-LayerNorm、GraphGPS 与 path MLP 可使 path logits 对 \(G\) 呈上下文依赖的非线性；alternative 的 `logsumexp` marginalization 可进一步使 \(\rho\) 弯折；gene softmax 则使 path \(\log P\) 与 \(P\) 弯折，但其公共 partition function 在 \(\rho\) 的两个 path-set log-masses 中严格抵消。上述非线性甚至可以非单调，但不等于模型学习了可独立解释的 factor/event-specific 剂量响应。当前 V2 不显式参数化或保证单调性、阈值、饱和常数、Hill coefficient、binding occupancy 或 cooperativity，也不支持把训练范围外的曲率解释为生化 kinetics。这个限制与“一层是否具有全基因感受野”是两个独立问题。

由于局部分支允许双向邻接且全局 attention 没有转录方向约束，正确表述是 `gene-level structural contextualization`，不是“信号沿 Pol II 方向传播”，更不是因果传播。若 event neutralization 对大量不共享 anchor region 的 paths 产生近似一致效应，应优先诊断 global-attention leakage 或 cell-state proxy。

### 8.3 Runtime unit

科学拓扑仍是每个 gene 一张共享 graph，但 cell-conditioned hidden state 与 forward unit 是唯一 `(gene, cell)` instance。训练 batch 按 gene 组织多个 cells；`edge_gene_index` 必须索引 gene-cell instances，绝不能让不同细胞之间共享 attention。属于同一 `(gene, cell)` 的多个 EC rows 只计算一次图和 path logits，再将结果广播到 compatible-set likelihood。

可缓存的 cell-independent 对象包括 graph topology、routing、ordered legal paths、path pooling index、attention mask、reporting/identifiability indices、静态 \(c_e\) 及其初始线性投影；cell-conditioned event aggregates 与 GraphGPS 输出 \(H_{i,g}\) 不得缓存为跨细胞共享值。batch size 按 graph edge 数动态控制。若 gene \(g\) 在当前 batch 含 \(B_g\) 个 cells 和 \(E_g\) 个 edge tokens，则 dense attention memory 量级为：

\[
O\left(\sum_g B_gE_g^2\right).
\]

## 9. Path-context readout

### 9.1 Path vector

上述 \(H_{i,g}\) 的每一 row 都是对应 processing-edge token 的 contextual state；后文以 \(h_{i,e}\) 专指其中 edge token \(e\) 的 row。V1/PRISM 的主要限制是先把每个局部 edge/alternative 标量化，再沿 path 线性求和。V2 改为先汇总每条合法 path 的 contextual vectors，但不使用 path mean：若两条等长 paths 只替换一个内部 token，mean difference 与该 token 的直接梯度都会被固定缩小为 \(1/|p|\)，从而对长 paths 形成不必要的内部剪接优化偏置。

令 \(I^{path}_{p,e}\in\{0,1\}\) 为去重后 structural path–edge incidence，且 \(\mathcal Y_g\) 只包含该基因冻结的唯一 structural paths。先定义不读取表达量、DTU 或 split outcome 的均匀 catalog incidence baseline：

\[
\bar I^{path}_{g,e}
=
\frac{1}{|\mathcal Y_g|}
\sum_{q\in\mathcal Y_g}I^{path}_{q,e}.
\]

同时定义不进入模型输入的 catalog complexity audit quantities：

\[
V_g
=
\sum_{e\in\mathcal E_g}
\mathbb 1[0<\bar I^{path}_{g,e}<1],
\qquad
D_g^{path}
=
\sum_{e\in\mathcal E_g}
\bar I^{path}_{g,e}(1-\bar I^{path}_{g,e}).
\]

\(V_g\) 是 variable-edge 数，\(D_g^{path}=|\mathcal Y_g|^{-1}\lVert I^{path}-\bar I^{path}\rVert_F^2\) 是 centered-incidence coefficient energy；二者仅用于 `PathScaleAudit`，不作为 path MLP 输入、样本权重或当前主模型缩放因子。

每条 path 使用固定的 gene-centered residual sum：

\[
\zeta^{path}_{i,p}
=
\sum_{e\in\mathcal E_g}
\left(I^{path}_{p,e}-\bar I^{path}_{g,e}\right)h_{i,e}.
\]

实现时不构造稠密 centered incidence。先用现有 sparse path incidence 计算 \(U^{path}_{i,p}=\sum_e I^{path}_{p,e}h_{i,e}\)，再计算：

\[
\zeta^{path}_{i,p}
=
U^{path}_{i,p}
-
\frac{1}{|\mathcal Y_g|}
\sum_{q\in\mathcal Y_g}U^{path}_{i,q}.
\]

该实现与上式严格等价，并保持 path aggregation 的稀疏边界。

因此任意两条 paths 满足：

\[
\zeta^{path}_{i,p}-\zeta^{path}_{i,q}
=
\sum_{e\in\mathcal E_g}
\left(I^{path}_{p,e}-I^{path}_{q,e}\right)h_{i,e}.
\]

两条 paths 只交换内部 tokens \(a,b\) 时，其 residual difference 直接为 \(h_{i,a}-h_{i,b}\)，不随共同 constitutive path 长度衰减；所有 paths 共有的 edge 因 \(\bar I^{path}_{g,e}=1\) 自动抵消。对于 path \(p\)：

\[
h^{path}_{i,p}
=
\operatorname{concat}\left[
\zeta^{path}_{i,p},
h_{i,first(p)},
h_{i,last(p)},
\log(1+|p|)
\right].
\]

`first` 和 `last` 按转录方向定义，固定字段 `log_edge_count` 的数值就是 \(\log(1+|p|)\)，显式保留路径长度信息。centered residual sum、first、last 与 length 均由现有冻结 path incidence 确定，不增加 learned pooling、reference transcript 或 choice-level training structure。V2 不另加 raw mean/sum 分支、独立 length penalty，也不使用 path ID、transcript ID 或 gene-specific free coefficient。

未缩放 residual sum 有意保留多个 variable choices 的累积证据，但其范数没有固定上界：在近似不相关的 token directions 下，典型平方范数可随 \(D_g^{path}\) 增长；高度同向时还可能更快。当前 V2 不直接除以 \(\sqrt{V_g}\)，因为这会让同一 focal exon swap 的差分随该 gene 其他无关 variable choices 再次缩小。`PathScaleAudit` 必须按 \(D_g^{path}\)、\(V_g\) 与 \(|\mathcal Y_g|\) 分层记录 gene-cell/path 的 \(\lVert\zeta^{path}\rVert_2\) median/q95/max、相对 token RMS、path-MLP 第一层 preactivation norm、path-logit SD/range、softmax entropy 与 compatible-set probability calibration、局部 contrast 对 \(\zeta^{path}\) 的梯度范数，以及 prediction/attribution seed stability。仅观察到 \(\lVert\zeta\rVert\) 随复杂度增长不构成失败；只有预冻结的 finite-output、compatible-set calibration、gradient-scale 或 attribution-stability criterion 在复杂 strata 中失败时，才限制该 criterion 所约束的 claim。这里的 `finite-output` 要求所有输入、输出和梯度均为有限值：出现未声明 NaN/Inf 会直接使对应 run/stratum evaluation invalid 并禁止其全部 prediction/mechanism claim，而不是只作幅度降级；数值仍有限但超出预冻结 range、gradient-scale 或 magnitude-stability criterion 时，才按实际 criterion 限制相应 claim，其中单纯 magnitude stability 失败只限制 path-specific magnitude claim。

若 synthetic/train/validation 在 held-out test model inference 首次运行前触发上述 scale-linked failure，唯一预声明的候选修订是先修改合同，再令

\[
\widetilde\zeta^{path}_{i,p}
=
\frac{\zeta^{path}_{i,p}}
{\sqrt{\max(1,D_g^{path})}}
\]

并由用户逐命令重训全部受影响的 conditions 与 seeds。该固定 catalog RMS scaling 不是最坏情况硬界，也不是 inference-time 开关；不能使用 test 结果选择是否启用。它保持共同 constitutive padding invariance，且同一 gene 内 path-pair difference 只被同一个正标量等比缩放。当前 `V2_DESIGN_FINAL` 的主模型仍使用未缩放 \(\zeta^{path}\)；LayerNorm、path-specific norm clipping 与 learned scale 不属于获准 fallback。

### 9.2 Shared path scorer

所有 genes 和 paths 共享一个小型两层 readout：

\[
s_{i,p}=\operatorname{MLP}_{path}(h^{path}_{i,p}).
\]

`MLP_path` 只包含一个 hidden layer、一个非线性和一个 scalar output。其容量在正式训练前冻结，不建立 readout registry、Transformer、RNN 或多种 pooling 分支。

非线性 path readout 允许同一 TSS、exon 或 PAS event 在不同完整路径上下文中产生不同甚至相反的边际效应，也允许更一般的 choice-context dependence。它不保证表达任意高阶依赖；这一能力必须通过合成数据和 held-out performance 验证，不能由网络容量直接宣称。

该 pooling 是有意选择的最小压缩表示，但不是数学上的 injective path encoding：\(p\ne q\) 不保证 \(h^{path}_{i,p}\ne h^{path}_{i,q}\)。matrix axis 中 ordered edge sequence 完全相同的重复 transcript 必须在进入模型前合并为同一 structural path identity。对监督可辨识的不同 paths，必须审计 pooled representations 是否发生精确或系统性近似 collision。长度缩放分成两个层次：固定 contextual states \(h\) 时，逐步向全部比较 paths 加入共同 constitutive tokens，\(\zeta^{path}_{i,p}-\zeta^{path}_{i,q}\) 及其对 differential tokens 的 Jacobian 必须严格不变，mean-only 负对照必须复现 \(1/|p|\) 衰减；完整 GraphGPS+MLP 的 relative log-mass 允许因 topology 与 `log_edge_count` 改变，只要求在合成和真实数据中按 path length、\(D_g^{path}\) 与 \(V_g\) 分层诊断，不宣称严格 padding-invariant。若真实数据仍出现系统性 collision、长度或 complexity-linked 失效，应停止对应 path-specific claim；不得静默加入 path-ID embedding、attention pooling 或 choice-specific parameter 来掩盖问题。

## 10. Legal-path probability 与 loss

只在同一基因的 matrix isoform paths 内归一化：

\[
P_{i,g}(p)
=
\frac{\exp s_{i,p}}
{\sum_{q\in\mathcal Y_g}\exp s_{i,q}}.
\]

对 \(k\in\mathcal K^{inf}_{i,g}\) 的 compatible set \(C_k\) 和 molecule weight \(n_k\)，split \(S\) 上的唯一数据拟合目标为：

\[
\mathcal L_S
=
-
\frac{
\sum_{(i,g)\in S}\sum_{k\in\mathcal K^{inf}_{i,g}} n_k
\log\left(\sum_{p\in C_k}P_{i,g}(p)\right)
}{
\sum_{(i,g)\in S}\sum_{k\in\mathcal K^{inf}_{i,g}}n_k
}.
\]

train、validation 与 test 分别使用各自冻结的 likelihood-informative molecule total 作为分母；同一 split 内 batch loss 的 numerator/denominator 必须按完整 epoch 累积，不能改成未加权的 batch mean。V2 不增加 choice auxiliary loss、DTU prediction head、triplet loss、contrastive loss、reconstruction loss 或 abundance MSE。§15.5 的参数收缩是对有限样本方差的优化器正则化，不是额外 biological target 或监督头。空 compatible set 与全 path compatible set 严格按 §3.3 处理，不能静默改变 likelihood 分母。

## 11. AlternativeReportingIndex

V2 不把 ChoiceCatalog 作为 trainable model object，但局部 alternative 仍需要一个冻结、可审计的 reporting index。它保留当前 elementary-choice 的 single-entry/single-exit、内部无再次分叉、内部 node-disjoint、non-overlap/non-nesting 检查，并增加 TSS、PAS、path-group、matched-context 与 supervision-identifiability 映射；它不产生训练参数或额外 loss。

TSS choice 按 path 的真实 TSS endpoint 分组，PAS choice 按真实 PAS endpoint 分组，internal choice 按确定性的 branch-reconvergence subpath 分组。对 internal choice \(c\)，eligible path 必须同时经过冻结的 entry 与 exit，并在二者之间唯一匹配恰好一个 alternative subpath；各 \(\mathcal Y_{c,a}\) 必须两两不交且其并集恰为 \(T_c\)。未同时经过 entry/exit、匹配零个或多个 alternatives 的 paths 从该 choice 的局部比较中排除，不能默认作为 `not_traversed` biological alternative。

令 choice \(c\) 有 \(A\) 个 alternatives，并令 \(T_c=\bigcup_a\mathcal Y_{c,a}\)。混合 traversing 与 non-traversing paths 的 EC rows 仍进入完整 path likelihood，但不计入 direct local support。局部可辨识性不得用 compatible set 内的 path-count fraction 近似，因为 compatible likelihood 并不假设组内 paths 等概率。

在 §3.4 的 \(m\) 个 observational groups 上，令 \(D_g^{train}=[\widetilde B^{train};\mathbf1^\top]\)。对每个 alternative 定义 group-subset indicator \(a^{c,a}\in\{0,1\}^m\)：若 \(E_h\subseteq\mathcal Y_{c,a}\) 则 \(a_h^{c,a}=1\)，若 \(E_h\cap\mathcal Y_{c,a}=\varnothing\) 则为零；若 group 跨越 alternative 或 traversing/non-traversing boundary，该 indicator 无定义并直接禁止正式 local attribution。alternative mass 只在其 indicator 属于完整 compatibility operator 的 row space 时可辨识：

\[
a^{c,a}\in\operatorname{row}(D_g^{train}).
\]

实现用 SVD pseudoinverse 检查：

\[
\left\|
a^{c,a}
-
(D_g^{train})^+D_g^{train}a^{c,a}
\right\|_2
\le
10^{-8}\max(1,\|a^{c,a}\|_2).
\]

只有 contrast 的 numerator 与 denominator subset indicators 都通过，才标为 `cohort_local_contrast_separable`；未属于这两个 subsets 的其他 paths 保留为 nuisance，不能先从 operator 中删除。另以一个保守但可审计的 direct-support gate 要求每个被比较 alternative 至少有一个 exclusive compatible molecule equivalent：

\[
S^{train}_{c,a}
=
\sum_{k\in\mathcal K^{inf}_{train,g}} n_k\,\mathbb 1[C_k\subseteq\mathcal Y_{c,a}]
\ge1.
\]

当前 cell 是否 `direct_cell_supported` 由 §3.4 从该 cell 原始 path-level \(B_{i,g}\) 的 group-constant rows 构造的 \(D_{i,g}=[\widetilde B_{i,g};\mathbf1^\top]\) 上重复同样的 row-space 与 exclusive-support 条件决定；`novel_split_group_row` 不参与该升级。只通过 train catalog 条件的 local result 必须标为 `cohort_identifiable_model_prediction`。该判据直接来自 compatible likelihood 对 subset probability 的线性可估性，不把其他 path probabilities 设为相等。

### 11.1 Marginal alternative contrast

令 \(\mathcal Y_{c,a}\) 表示在 choice \(c\) 中采用 alternative \(a\) 的合法 paths。对于 alternatives \(a,b\)，总体 path-marginal alternative relative log-mass 定义为：

\[
\rho_i(c,a:b)
=
\operatorname{logsumexp}_{p\in\mathcal Y_{c,a}}s_{i,p}
-
\operatorname{logsumexp}_{p\in\mathcal Y_{c,b}}s_{i,p}.
\]

该量是 gauge-invariant 的总体路径质量对比，但它可能同时包含其他 TSS、exon 与 PAS context 的差异，不是模型内部原生的“局部剪接能量”。对于 one-versus-rest，第二项必须是所有其他 eligible alternatives 的 path complement；用全部 eligible paths（包含 \(a\)）作第二项只得到 conditional log probability，不能称为 logit。

### 11.2 Matched-context alternative contrast

为得到更接近“只改变当前 choice”的局部比较，reporting index 使用 matrix isoform 结构、且不读取 outcome，为每条 path 构造 choice 外部 context signature \(\kappa_c(p)\)。internal choice 删除 entry–exit alternative subpath 后，剩余 transcript-oriented prefix 与 suffix 必须相同；TSS/PAS choice 使用相应共享 downstream/upstream skeleton。定义：

\[
\mathcal Y_{c,a,m}
=
\left\{
p\in\mathcal Y_{c,a}:
\kappa_c(p)=m
\right\}.
\]

当 alternatives \(a,b\) 在同一 context \(m\) 中均存在合法 paths 时：

\[
\rho_i(c,a:b\mid m)
=
\operatorname{logsumexp}_{p\in\mathcal Y_{c,a,m}}s_{i,p}
-
\operatorname{logsumexp}_{p\in\mathcal Y_{c,b,m}}s_{i,p}.
\]

matched-context contrast 是机制解释的首选；marginal contrast 用于描述总体路径重分配，两者同时保留、不得混名。对固定 context \(m\)，把 subset indicators 改为 \(\mathcal Y_{c,a,m}\)，逐字重复上述 full-operator row-space 与 exclusive-support 检查。任一 observational-equivalence group 若跨越 \((a,m)\) partition，即使始终属于同一 alternative，也禁止对应 matched-context attribution。V2 第一版只报告 context-specific \(\rho_i(c,a:b\mid m)\)，不定义或报告跨 contexts 的加权汇总。复杂、嵌套或重叠区域若无法得到无歧义 path grouping，仍参与完整 path prediction，但不得强行报告局部 alternative effect。

`AlternativeReportingIndex` 必须同时冻结 matched-context applicability，而不能只列出已经被 explanation manifest 选中的 records。alternative pair 以 unordered canonical pair 或等价的冻结 `contrast_id` 唯一计数，不能同时把 \(a:b\) 与 \(b:a\) 算作两个 candidates。对每个至少含两个 alternatives 且通过本节结构规则的 choice，保存 `n_matched_context_candidates`，即唯一 alternative pair × context 中两臂 path subsets 均非空的数量，以及 `has_matched_context_structure`；联结只由 train supervision 构建的 row-space/exclusive-support status 后，再保存 `n_cohort_reportable_matched` 与 `has_cohort_reportable_matched`。evaluation 时另计算 direct-cell-supported records，最后才与 explanation-manifest selection 相交。choice-level coverage waterfall 固定为 `all structurally valid choices -> has >=1 two-arm matched context -> has >=1 cohort-reportable matched contrast -> has >=1 direct-cell-supported held-out record -> has >=1 held-out record that is both direct-cell-supported and explanation-manifest-selected`。record-level coverage 在每个 split 内以全部 model-eligible `(cell, gene)` 与该 choice 的 split-neutral two-arm `(contrast_id, context)` candidates 的笛卡尔积为固定分母，再依次报告 `cohort-reportable -> direct-cell-supported -> both direct-cell-supported and manifest-selected`；分母为零标记 `not_estimable`，不能填零，也不能在同一比例中途更换统计单位。至少按 TSS、internal 和 PAS 报 choice-level count/fraction，并另报 pair×context 与 cell-record counts；结构层是 split-neutral，cohort 层是 train-only，direct-cell 层是 evaluation-only，manifest 层是预声明的计算子集，不能混称为同一种 coverage。

## 12. Attribution semantics

### 12.1 Evidence neutralization

令 \(S\) 为预先定义的 evidence selector，并令 \(\mathcal R(S)\) 是它确定的 route set。\(X^{(-S)}\) 表示在 event aggregation 前仅将 \(r\in\mathcal R(S)\) 的 routed terms 置零，其余输入保持不变：

```text
full evidence:        retain all R[r,e] * G[i,tau(j(r))]
                      * (W_base*u_base[r] + W_int*(m_int*u_int[r]))
neutralized evidence: set routed terms with r in Routes(S) to zero
```

V2 固定支持三种 selector，且映射唯一：单物理 event selector 映射到该 event 的全部 routes；factor/group selector 映射到该 entity 全部 motif events 的全部 routes；anchor-region selector 只映射到 `anchor_region_id` 相同的 DNA/RNA routes。它们分别回答 event、factor/entity 和局部 regulatory neighborhood 在当前背景下的模型边际效应，不新增训练头。因而 anchor-region neutralization 不会删除同一物理 event 在其他 regions 的 routes；event/factor neutralization 则不会遗漏其多 route evidence。selector ID、解析出的 route IDs 与 route count 都必须随 attribution 保存。

对 gate-key component \(C\)，§6.3 的 `correlated_evidence_set` neutralization 唯一定义为 \(\mathcal R(C)=\{r:\tau(j(r))\in C\}\) 的去重 route union；不得展开为会误删 component 外同 factor peaks/events 的完整 factor selector。它不是第四种可学习 selector，也不改变任何 event identity；manifest 必须保存 component ID、成员 gate keys、对应 events 和完整 route union。该联合效应仍是 set-level 模型反事实，不得称为已辨识的 group contribution。

§5.5 的 multi-member model-injection equivalence class \(E\) 以 \(\mathcal R(E)=\bigcup_{j\in E}\mathcal R(j)\) 定义 reporting-only derived selector。正式 attribution 必须联合 neutralize 完整 union 并实际重新 forward；不得挑选 representative member，也不得用成员数乘单 member effect。单 member neutralization只作 `exchangeable_member_counterfactual` 诊断。任何 primitive selector 或 correlated-evidence selector 若只覆盖某个 equivalence class 的真子集，必须标记 `partial_model_injection_group`；该结果不得进入 primary mechanism、seed stability、between-state 或 null-attribution summary。只有完整 injection classes 的 union 才具有正式 attribution scope。

neutralization 数值上允许在缺失动态证据时得到零，但缺失不等于“观测到零效应”。对当前 cell 的 attribution，RNA motif event 只有在所需 \(m^F_{i,h}=1\) 时、factor-specific DNA event 只有在 \(m^F_{i,h}=m^A_i=1\) 时、accessibility-only event 只有在 \(m^A_i=1\) 时，才具有可解释的当前动态 context。selector 含多个 events/routes 时，其全部所需 masks 都必须有效；否则该 record 标为 `missing_context_not_estimable`，仍可保留完整模型 prediction，但不计算或不汇入 primary event-mechanism effect summary。

对任意输出 \(Q\)，定义 \(\Delta_SQ=Q(X)-Q(X^{(-S)})\)。主要报告：

\[
\Delta_S\rho_i(c,a:b)
=
\rho_i^{full}(c,a:b)
-
\rho_i^{(-S)}(c,a:b),
\]

\[
\Delta_S\log P_{i,g}(p)
=
\log P^{full}_{i,g}(p)
-
\log P^{(-S)}_{i,g}(p),
\]

\[
\Delta_SP_{i,g}(p)
=
P^{full}_{i,g}(p)
-
P^{(-S)}_{i,g}(p).
\]

对于 supervision-unidentifiable group \(E\)，path-specific 数值仅作模型诊断，主要输出改为 \(\Delta_S\log P_{i,g}(E)\) 与 \(\Delta_SP_{i,g}(E)\)。gene softmax 对 path logits 的共同平移不敏感，因此裸的 \(\Delta s_{i,p}\) 不是主要 estimand。若保留 logit change，唯一允许的 gauge fixing 是对全部 unique structural paths 等权中心化：

\[
\Delta_S\widetilde s_{i,p}
=
\Delta_Ss_{i,p}
-
\frac{1}{|\mathcal Y_g|}
\sum_{q\in\mathcal Y_g}\Delta_Ss_{i,q}.
\]

不得改用 probability weighting 或 equivalence-group weighting 而不更名 estimand。`Δlog path/group probability`、`Δpath/group probability` 和 \(\Delta\rho\) 是主要 gauge-invariant 输出。

### 12.2 Source-proxy and count perturbation

`source_proxy_perturbation` 直接修改某 activity entity 的 \(F_{i,h}\) 或指定 peak(s) 的 mapped \(A_{i,p}\)，再使用冻结的 train \(\mu/\sigma\) 重算全部受影响 gates；ATAC proxy 改变必须同时更新共享这些 peaks 的 factor-specific 与 accessibility-only events。除非实验本身改变 neighbor mapping 或其 QC，\(m_i^A\) 与只作报告的 neighbor-support metadata 保持不变。`member_count_perturbation` 则修改 group member 或 unique factor 的原始 \(x_{i,f}\)，按 §6.1 重算所有包含该 gene 的 activity proxies 与 gates。两者都不得只把 final gate 设为零，也不得只删除任意一个 motif hit。每次预测必须列出 perturbation kind 与输入值、受影响 gate 的 train min/max 与 frozen quantile interval、raw signal \(b\)、standardized residual \(z\)、final gate \(G\)、observation masks、两个 out-of-support flags、全部受影响 events、local contrasts 与完整 path/groups。任何超支持结果必须标为 `model_extrapolation`，不得混入 primary supported mechanism summary。

对同一观测 cell 的 in-silico `member_count_perturbation`，CP10K denominator \(L_i\) 固定为原观测 library denominator，只重算受影响 unique/group numerators，避免一个 factor perturbation 通过 compositional renormalization 改变所有无关 entities。若输入是独立测得的真实 perturbation library，则先用该 library 自身的完整 counts 与 denominator 重新构造所有 proxies，再作为新的 `observed_library_context`；三种语义必须分开命名。

如果报告 source-proxy 或 member-count 的 response curve，sweep 必须从所声明的干预层级开始：\(F_{i,h}\) 与 mapped \(A_{i,p}\) 已是 source proxies，只重算其下游 raw signals、冻结 \(\mu/\sigma\)、masks 与 gates；\(x_{i,f}\) 则必须先按 §6.1 重算 CP10K-log1p activity proxy，再重算全部下游量。不得直接扫描 final gate 后把横轴称为 factor concentration。每个点必须列出全部受影响 gate keys 与 events，而不是只显示一个被挑选的 motif hit；与当前 claim 相关的全部 affected raw signals 都必须落在各自冻结的 train min/max 和 quantile support 内，否则该 curve point/record 标为 `model_extrapolation`，不得进入 supported association summary。response curve 只描述固定模型的预测敏感性，不得从其形状声称 Hill kinetics、occupancy、阈值或 cooperativity。

evidence neutralization 是模型输入基线反事实；source-proxy/member-count perturbation 是对观测输入的模型反事实或外推；`observed_library_context` 是一个新的实测 context；基因组 motif deletion 还需要同步修改 CIS/sequence 表示，当前 V2 不模拟。它们不得混称为 knock-out 或因果干预。

### 12.3 Explanation manifest and non-additivity

全部 eligible cell-gene instances 都参与 prediction 与 NLL，但昂贵的逐 event 重算只在预声明的 explanation manifest 上运行。manifest 必须冻结 `cells/cell states`、`genes`、`events/factor groups/anchor regions`、完整 `model_injection_group_id/member_event_ids`、`alternative contrasts` 和 `path/equivalence groups`。选择只能依据 split-neutral QC、train/validation support、预声明 case genes 与 matched controls，不能依据 test effect magnitude，也不能从 injection class 内选择 effect 最大的 member；结果必须报告 manifest 规则和相对于全部 eligible 单元的覆盖率。

由于 event blocks 共同经过 GraphGPS 和非线性 path readout，单事件 effect 是“当前其他 events 与当前细胞背景下的模型反事实边际效应”。多个单事件 effects 不保证精确相加，也不构成唯一机制分解。V2 第一版不计算 Shapley values，不把 attention weight 当作 contribution，也不把隐式 DNA×RNA 或 event×event statistical interaction 解释为蛋白物理互作。

每条 attribution record 至少保存：

```text
cell_id and reporting cell state
gene_id
event_id or event_set_id
factor_entity_id, identity kind, group and candidate factors
activity_entity_id, activity_gene_ids and self-factor flag
motif_id
motif_equivalence_family_id and source_motif_ids
coordinate and strand
DNA/RNA channel and event kind
anchor region and complete routes
current-cell ATAC mapping_valid, valid-neighbor count K_i, ESS, weight evenness/coverage, distances and consistency status, when applicable
current-cell required observation masks mF/mA and context estimability
raw signal b, standardized residual z, final gate G and out-of-train-support flags
attribution_scope = supported_context | model_extrapolation | missing_context_not_estimable
linked raw-interaction contrast records, one per predeclared contrast when applicable
raw_interaction_claim_status = factor_specific_grammar_estimable | cross_field_context_not_separable | within_factor_only | raw_contrast_not_in_active_span | unsupported_focal_arms | not_applicable_open_only | not_applicable_no_predeclared_contrast
raw_interaction_contrast_id = (modality, context_field, factor_entity_id, context_level_pair)
raw_interaction_comparator_ids_that_pass_active_span and per-comparator contrast_in_active_span
cross_field_context_separable and aliased_context_field_ids
encoded_interaction_basis_status and rank-audit status
model_injection_group_id, member_event_ids and member_count
model_injection_scope = singleton_supported | set_supported | exchangeable_member_diagnostic | partial_model_injection_group | not_applicable_modality_absent
model_injection footprint_relation and motif_family_relation
evidence_separation_status, pairwise correlated gate keys and correlated_evidence_set ID, when applicable
choice, contrast kind and matched-context signature
path_id or observational-equivalence-group ID
support_tier = direct_cell_supported | cohort_identifiable_model_prediction | supervision_unidentifiable_prediction
per-seed and across-seed median/IQR/sign agreement for delta local relative log-mass
per-seed and across-seed median/IQR/sign agreement for delta path/group log probability
per-seed and across-seed median/IQR/sign agreement for delta path/group probability
```

## 13. Cell state policy

V2 主模型删除独立 `StateScorer`。细胞状态通过当前细胞实测或映射得到的 factor activity、RBP activity 和 ATAC context 条件化；stage、developmental system、cell type 和 donor labels 不作为第四个模型 block，而用于 split、stratified reporting、matched null 和生物学解释。

删除 free State branch 是一个明确科学赌注：motif catalog 未覆盖的 factor、翻译后修饰、间接调控和未测量状态效应会成为不可约误差；同时，相关 factor activities 也可能代理整个 cell state。为此，§16.4 的 held-out state-residual gate 是正式机制 claim 的强制 admission 条件。加入 nuisance-state baseline 需要单独的架构修订，不能在训练过程中静默恢复 `StateScorer`。

## 14. High-DTU policy

high-DTU 不是模型结构、预测头或新 loss。主分析保留完整 eligible gene cohort，并保留 low-DTU/stable genes 作为必要对照；不得只选择 high-DTU genes 形成唯一训练或测试 universe。high-DTU 信息只允许用于：

1. 单独标记的 train-only、预先有界的 sampling/weighting sensitivity；
2. validation/test 的预声明分层评价；
3. 分析 DNA/RNA 增益是否集中在真正 context-responsive genes。

`data/DTU_score.R` 以预先存在的 190-cell-type PSI/表达矩阵为输入，在 PSI、表达和 transcript-to-gene metadata 的共同 transcript 轴上要求至少两条 transcripts、至少两个 expressed cell types，并以 dominant-transcript switching 的 transcript-wise JS divergence 计算 score。交付的 `data/DTU_result_sorted.xlsx` 与 matrix-matched GTF 的 28,002-gene transcript counts 完全一致；其中 `top_DTU_gene` 恰好等价于 `DTU_score >= 0.7`，但阈值赋值代码不在该 R 文件中，且输入聚合是否包含 held-out test cells 尚无 split provenance 记录。因此该 label 当前只能作为 external diagnostic stratum，不得进入训练采样、权重或 gene admission；primary training 使用下一节定义的未重加权全 cohort objective。未来任何 train-derived high-DTU 定义只能在 train split 内拟合，再冻结到 validation/test。

## 15. Training design

### 15.1 Per-command conditions

训练 runtime 只暴露三个可独立选择的 condition：

- `full`：CIS + DNA/ATAC + RNA/RBP；
- `atac`：CIS + DNA/ATAC，RNA/RBP block 在进入 GraphGPS 前置零；
- `rbp`：CIS + RNA/RBP，DNA/ATAC block 在进入 GraphGPS 前置零。

每条命令恰好选择一个 `--seed` 和一个 `--condition`，只训练一个模型。合同不再定义 exploratory/formal 两套 full-cohort runtime，不要求一条命令自动展开多个 conditions 或 seeds，也不要求用户先声明一个固定 campaign。需要重复 seed 时，用户分别提交多条命令；需要比较 `full`、`atac`、`rbp` 时也分别提交，并在作配对比较时自行保持 graph、path catalog、split、loss、optimizer、gene-cell cap 与其他配置一致。

三个 condition 始终实例化完全相同的 \([CIS\mid DNA\mid RNA]\) 输入宽度、\(W_D^{base/int}/W_R^{base/int}\)、canonical interaction active-basis masks、GraphGPS、path MLP tensor shapes、hidden widths 和总参数量。未使用的模态只在进入 GraphGPS 前硬置零，不删除输入列、不缩小投影或 backbone。命令 seed 同时决定模型初始化和 §15.2 的逐 epoch 确定性 gene-cell 重采样；相同 seed、condition、配置和输入必须可复现相同初始化与采样序列。

CIS-only 与 `Full-AdditiveEdgeReadout` 不属于训练 CLI condition，也不构成另一套 runtime。若以后需要 architecture comparator、额外消融或批量 campaign，只能作为显式、独立的分析工作流逐条调用同一单任务 runtime；不得重新在 `train.py` 内硬编码 multi-run matrix。Full 内部的 accessibility-only、TF-DNA、单 factor/entity、单 motif event、anchor region 或单 channel neutralization仍是 inference-time evidence analysis，不是额外 trainable heads。

### 15.2 Sampling unit and primary objective

primary train universe 是全部具有正 likelihood-informative molecule mass \(w_{i,g}=\sum_{k\in\mathcal K^{inf}_{i,g}}n_k\) 的 eligible \((gene,cell)\) instances，但所有 V2 训练模式都用相同的 per-gene capped stochastic epoch，而不在单个 epoch 穷举该 universe。sampling unit 固定为完整 \((gene,cell)\) group；同一 instance 的全部 informative EC rows、ordered compatible sets 与原始整数 molecule counts 必须一起进入 loss，audit-only rows 随 provenance 保存但不进入 objective，禁止独立截断 EC rows 或把同一 cell-gene 拆成重复 examples。

配置中的 `max_train_gene_cells_per_gene_per_epoch` 是正整数运行参数；512、1024、2048 等值均合法，并写入该命令的 `TrainingRunManifest`。每个 epoch、每个 gene 从其 \(N_g\) 个 train gene-cells 中按冻结 G_fit order、命令 seed 与 epoch 确定性地无放回均匀抽取 \(n_g=\min(N_g,\text{cap})\) 个；逐 epoch 重新抽样，validation 始终完整遍历且不应用该 cap。不存在独立的 `sampling_seed` 配置字段，也不存在 runtime 自动生成的多 seed/多 condition campaign。若多个独立命令要作严格配对比较，用户必须显式保持 cap、seed、抽样规则和所有其他非 condition 配置一致。

令被选 gene-cell 的完整 likelihood numerator 为 \(A_{i,g}\)，\(G=|G_{fit}|\)，\(M_{train}\) 为冻结的完整 train informative molecule total。optimizer step unit 固定为一个 train-positive gene：每个 epoch 将全部 \(G\) 个 genes 确定性打乱后各执行恰好一次 `optimizer.step()`。一个 gene 因 GPU batch policy 拆出的多个 cell microbatches 只依次 `backward()` 并累积梯度，在该 gene 的全部 sampled cells 完成前禁止 step；因此 packing 不改变 update count。

GPU packing 固定为 `gene_shape_adaptive_v1`，不再用单一 `cells × edge_count²` attention cap。每个 gene 分别以 routes、edges、paths、compatible width、实际 EC-row counts 和冻结 model widths 形成 static/per-cell/per-compatible-row shape cost；train 与 evaluation 使用各自冻结的 bytes-per-shape-element 系数，在 `target_gpu_allocated_bytes` 减去 `unmodeled_gpu_reserve_bytes` 后确定最大完整 cell-group batch，同时受明确的 `max_cells_per_gpu_batch` CUDA kernel-shape 上限约束。planner 只依赖冻结 shape/config，不读取瞬时 free-memory、不捕获 OOM 后重试、不拆散同一 gene-cell 的 EC rows。full-cohort runtime 启动时必须确认单张已 profile GPU 的 total/free memory 能覆盖目标；否则明确失败。backed dataset 可用一个 CPU worker 预取严格 gene order 中的下一 immutable shard，但 model forward、RNG、sampling、backward 与 optimizer step 仍在原调用线程按原顺序执行。compute precision 冻结为 `float32_highest` 且 CUDA matmul TF32 关闭；任何 mixed-precision/TF32 提速必须先通过独立数值等价审计，不能由 profiler 静默开启。

第 \(g\) 个 gene step 使用的 loss 为

\[
\widehat{\mathcal L}^{step}_g
=
\frac{G}{M_{train}}\frac{N_g}{n_g}
\sum_{i\in S_g}A_{i,g}.
\]

其中 \(N_g/n_g\) 校正 gene 内均匀无放回抽样，\(G\) 校正 uniform gene sampling 的 \(1/G\) 概率。在固定参数 \(\theta\) 下，对均匀抽取的 gene 与 gene-cell sample 取期望可恢复下面 full-cohort molecule-weighted objective 的梯度；runtime 使用无放回 random reshuffling 并在 genes 之间更新参数，因此整个 epoch 不冒充一次 full-batch gradient，也不声称每个有条件的 reshuffling step 都严格无偏。它不是 gene-equal objective，也不使用 sampled-mass denominator。省略 \(G\) 虽不改变 genes 间相对权重，却会把每步数据梯度整体缩小 \(1/G\)，并改变 AdamW epsilon 与 decoupled weight decay 相对数据梯度的尺度，因此禁止省略。

对应的完整 primary objective 是 §10 中 \(S=train\) 的完整-cohort、固定 molecule-total compatible-path NLL：

\[
\mathcal L_{train}
=
-
\frac{
\sum_{(i,g)\in train}\sum_{k\in\mathcal K^{inf}_{i,g}}n_k
\log\left(\sum_{p\in C_k}P_{i,g}(p)\right)
}{
\sum_{(i,g)\in train}\sum_{k\in\mathcal K^{inf}_{i,g}}n_k
}.
\]

因此 cap、逐 epoch resampling 与 randomized gene order 改变计算预算和随机梯度方差，gene-shape packing 只改变 gene 内 microbatch 数；它们均不改变目标 estimand。一个 epoch 恰好有 \(G\) 个 optimizer steps，而不是一个 epoch 一步，也不是每个 microbatch 一步。若 provenance 审计完成后运行 high-DTU oversampling 或额外 gene multiplier，它必须成为单独命名的 sensitivity objective，在上述 baseline inclusion probability 之外记录额外 sampling probability 或 multiplier，并始终在原始未重加权 test distribution 上报告 NLL 与 calibration；不得将其结果冒充上述 primary objective。

### 15.3 Split and claim scope

当前 primary split 直接采用 `docs/FABRIC_CELL_GENE_SELECTION.md` 已冻结的规则：217,933 个 ONT cells 在 Emb01--Emb09 每个胚胎内部独立执行 deterministic cell-level 80/10/10，split seed 为 `20260725`。划分单位必须是 globally unique cell ID，使同一细胞的所有 genes、molecules、matrix counts 和 EC rows 进入同一 split。它只支持九个已观测胚胎内部的 transductive supervised cell generalization；embryo holdout、donor holdout 或其他跨个体 claim 必须在独立 split 上验证，不能由该 cell holdout 结果代替。

split-neutral 对象仅包括冻结的 matrix transcript identity 及其 graph/structural paths、reference sequence、motif libraries、静态 event collapse/routing/cap、raw factor/context vocabularies、scientific context-pair universe、feature vocabulary和train-independent feature calibration；它们对所有 splits共用。train-only对象包括 CIS normalization、gate baselines/scales/admission/support、event `model_active`、raw interaction-cell support、supported-rectangle canonical basis、combined rank audit/final active-basis mask、raw contrast/comparator claim table、Path/Alternative IdentifiabilityIndex，以及任何获准的 high-DTU sampling/weighting sensitivity。validation NLL只用于 early stopping、优化/容量选择、预声明诊断参数选择和 explanation-manifest中允许的support筛选；不能重估前述train-only对象，§15.4 的 ONT fields 另受更严格的非选择边界约束。test model inference 与预冻结的评估/claim-admission 套件只运行一次，不产生任何模型、catalog、threshold、case或报告规则反馈。

当前 10% test cell 的 ONT count columns 已被 ONT-first gene-selection workflow 随完整 matrix 一起物化，并已发布 full-matrix aggregate count；gene admission 本身只使用 train columns，但该 test 不能再称为 `unopened` 或 fully blinded。它仍是固定、未参与模型拟合和 admission 的 held-out evaluation split，正式 checkpoint 后可以按预冻结规则一次性评价；任何结果只能明确称为 `previously_materialized held-out test`。若要支持真正 blind confirmatory test claim，必须使用此前未读取的新独立 cohort 或预先封存的数据，不能靠重新命名当前 split 恢复盲态。

### 15.4 Per-epoch ONT matrix-agreement monitor

每个 completed epoch 之后只执行一次完整 validation traversal，不再执行完整 train evaluation。该 traversal 在同一冻结 model state、`eval`/`no_grad` 下产生且只产生两个核心数值：

1. `validation_compatible_path_nll`：主要指标；唯一用于 early stopping 和 checkpoint 选择；
2. `ont_matrix_kl_count_weighted`：ONT 分布一致性诊断；sealed、selection-ineligible，只作报告。

两者复用同一 validation predictions；ONT 计算不得触发第二次 model forward。训练循环中已经计算的 sampled losses 不导出为 `train_nll`，因为它们跨越连续更新的参数状态，不能冒充一个固定 epoch-end model 的完整 train NLL。最佳 checkpoint 确定后也不再为生成 train metric 而完整遍历 train。

ONT target 只包含 17,600 个 G_fit genes 的完整 ordered model-path axis 与 21,788 个 validation cells；不得包含 test cells/counts。对 validation likelihood-informative \((i,g)\)，令 \(x_{i,g,t}\ge0\) 为原始 ONT matrix count，\(N_{i,g}=\sum_t x_{i,g,t}\)。冻结的普通 exclusions 仍只有 `ont_count_total_zero` 与 `fewer_than_two_positive_matrix_transcripts`；其余进入 eligible scope。path、gene、cell、split 任一 identity 不一致必须明确失败，不能取交集。eligible count、ONT count denominator 与两个 exclusion counts 是审计分母，不是额外评价指标。

对 eligible \((i,g)\)，定义：

\[
q_{i,g,t}=\frac{x_{i,g,t}}{N_{i,g}},
\qquad
\ell_{i,g,t}=\log P_{i,g}\!\left(\phi_g(t)\right),
\qquad
p_{i,g,t}=P_{i,g}\!\left(\phi_g(t)\right).
\]

\[
D_{KL}(q_{i,g}\Vert p_{i,g})
=
\sum_{t:q_{i,g,t}>0}q_{i,g,t}
\log\frac{q_{i,g,t}}{p_{i,g,t}},
\]

并按 ONT count 加权聚合：

\[
\operatorname{ont\_matrix\_kl\_count\_weighted}
=
\frac{\sum_{(i,g)\in eligible}N_{i,g}D_{KL}(q_{i,g}\Vert p_{i,g})}
{\sum_{(i,g)\in eligible}N_{i,g}}.
\]

实现必须直接从完整 path logits 以 float64 `log_softmax` 计算，不做 probability clamp。per-epoch runtime 不再计算或保存 ONT top-1/tie-aware accuracy、top-5、singleton hit、macro KL、cross-entropy、PRISM-clamped compatibility fields 或 compatible-set accuracy-like fields。因为 ONT matrix quantifier 与 compatible-read likelihood 尚不能证明共享完全相同的 observation population，该 KL 仍命名为 `same_library_cross_pipeline_ont_matrix_agreement`，不是 ground-truth accuracy，也不参与模型选择。训练期间不读取或计算 test predictions/metrics。

### 15.5 Optimizer and hierarchical shrinkage

稀有 factor×context 列不能依靠无收缩的最大似然估计。每条命令使用 AdamW 参数分组：bias 与 normalization 参数不做 weight decay；其他 base/backbone/readout weight matrices 使用一个共享的非负 \(\lambda_{base}\)；\(W_D^{int}\) 与 \(W_R^{int}\) 使用同一个更强且严格为正的 \(\lambda_{int}\)，并要求 \(\lambda_{int}>\lambda_{base}\ge0\)。这把显式 factor-specific positional deviation 向 `factor baseline + shared route-context baseline` 收缩，而不是把低支持 factor 合并为 `OTHER`；它不声称消除后续非线性网络可能形成的所有隐式 interaction。DNA/RNA 不分别设置 \(\lambda_{int}\)，不同 factors、genes 或 context bins 也不得拥有各自 penalty。

learning rate、scheduler 与 global gradient-clipping norm 是每条命令的显式运行参数，而不是代码常量。当前 runtime 只实现窄而明确的 `constant` 与 `reduce_on_plateau` 两种 scheduler；后者在每个 completed epoch 的唯一完整 validation traversal 之后恰好调用一次，只读取 `validation_compatible_path_nll`，不得读取 `ont_matrix_kl_count_weighted` 或 test。每个 gene 的全部 adaptive microbatches 必须先完成 backward accumulation，再对全模型梯度执行至多一次预声明 global-norm clipping，随后恰好一次 `optimizer.step()`；不能按 microbatch clipping 或 step。history 必须同时记录本 epoch 实际 learning rate 与 scheduler 更新后的下一 epoch learning rate。checkpoint 必须保存选中 epoch 的 model、optimizer 和（若启用）scheduler state，避免状态身份丢失。

本次待启动 `full` 命令的配置默认值为 `learning_rate=5e-5`、`reduce_on_plateau(factor=0.5, patience=1, min_lr=1e-5)` 与 `gradient_clip_norm=1.0`。这些值是启动前预声明的单次运行策略，不是 tuning 结果；CLI 可逐命令覆盖并把 resolved values 冻结进 `TrainingRunManifest`、resolved config、optimizer manifest、checkpoint 与 epoch history。参数化不授权自动 grid、multi-seed expansion 或 test-based selection。

\(\lambda_{base}\) 与 \(\lambda_{int}\) 是该命令所读取配置中的显式参数，不由训练 CLI 自动调参，也不要求先运行 grid、tuning seeds 或多 condition selection。用户可以在启动前选择任意满足上述不等式的 pair；若要比较不同 penalties，必须把每个 pair 当作独立运行并如实记录，不能根据 test 结果回选。checkpoint metadata 必须保存两个数值及每个参数所属分组；不得同时再对同一 interaction weights 叠加未记录的显式 L2 penalty。该收缩不修复秩缺失，因此 §5.5 的 raw-cell support、canonical support-span basis 和 combined-design rank audit 必须先执行。

第一版不使用 group lasso、稀疏硬选择的训练期 penalty、Bayesian hierarchy、horseshoe、per-factor regularizer 或 learned shrinkage network。这些机制不是当前合同所需。

### 15.6 Unified full-cohort runtime

唯一真实数据运行 scope 是 `full_cohort`。它绑定 train-only admission 后的全部 17,600 个 \(G_{fit}\) genes，并同时保留 17,706 structural candidates、90,672 paths、217,933 cells 和 106 个 graph-only genes 的完整 audit；不得用历史 7,198-gene graph/EC 或 167,235-cell split。当前唯一 PreparedDataset 位于 `data/processed/fabric_v2_real_dataset_v1/prepared_dataset/`：包含 17,600 个有序 G_fit shards、90,361 条 G_fit paths、493,310 条 G_fit edges、74,156,703 条 train/validation compatible rows 和 69,027,635 个 gene-cell instances。106 个 graph-only genes 独立保留且不进入 likelihood fit；test compatible rows 为 0。

一次启动只产生一个 `TrainingRunManifest`，其运行身份由命令行 `--seed <int>`、`--condition {full,atac,rbp}` 与 resolved config 共同决定。seed 与 condition 不写入 YAML，避免配置暗中展开 run matrix；同一命令 seed 同时是初始化 seed 与 gene-cell sampling seed。配置中的 `max_train_gene_cells_per_gene_per_epoch` 可设为 512、1024、2048 或其他正整数；每个 gene 最多抽取该数量的完整 train gene-cell groups，validation 始终全量遍历。CLI 可显式覆盖 learning rate、scheduler、scheduler parameters、gradient clip norm、penalties、gene-cell cap、max epochs 与 early-stopping patience；未覆盖字段读取 YAML。manifest 必须冻结全部 resolved optimizer/training controls、`optimizer_step_unit=train_positive_gene`、`gene_microbatch_gradient_accumulation=true`、`gene_shape_adaptive_v1` 显存目标与 kernel cell cap。不同命令可以选择不同值，但只有全部比较身份相同时才可作严格配对比较。

训练授权只有统一的 `execution.training_authorized` 布尔门，不存在 exploratory 与 formal 两种训练授权。held-out test 仍由独立的 `execution.final_test_authorized` 控制；训练期间只允许 train optimizer update 与完整 validation evaluation，不读取或生成 test compatible rows、test predictions 或 test metrics。测试、真实数据 validation、full-shape profiling、资源冻结与训练授权是不同状态，必须分别报告。

每个 completed epoch（该 epoch 的全部 gene updates、唯一完整 validation traversal、validation-NLL scheduler step 和 reporting-only ONT KL 均已完成）形成唯一恢复边界。runtime 必须以原子文件替换写入 `latest.pt`；它自包含当前 model/optimizer/scheduler、已完成 epoch、early-stopping counter、完整 history/monitor、gene-order RNG、Python/NumPy/Torch/CUDA RNG，以及当前 validation-NLL best 的 model/optimizer/scheduler state。`best.pt` 只在 validation NLL 严格改善时更新，训练正常结束后 `checkpoint.pt` 与 `best.pt` 均表示选中 best，而 `latest.pt` 保留最后 completed epoch 以供故障恢复。恢复只允许显式 `--resume-from <原 run-dir>/latest.pt` 并复用原 run-dir；seed、condition、resolved config、TrainingRunManifest、input/compatible-artifact identity、有序 G_fit、model shape 或 test-exposure 状态任一不一致均明确失败。恢复必须从 `completed_epoch+1` 开始，恢复 scheduler、early stopping、history 和全部 RNG 连续性；不得从 `best.pt` 分叉 history，不支持 gene 中途或 epoch 中途续跑，也不得因恢复读取 test。run-dir 使用进程互斥锁，禁止两个 trainer 同时改写同一恢复历史；若进程在 epoch 中途退出，磁盘上的上一份原子 `latest.pt` 仍是唯一合法恢复点。

截至 2026-08-14，旧的固定 attention cap 与 20.23-hour epoch 外推已经删除。当前 clean `ResourceProfile.json` 使用一张独占 RTX 4090、`full`、cap 512、一次完整 validation、两个 backed-shard traversals和单 worker prefetch：17,600 个 capped train genes 全部规划为每 gene 一个 GPU batch，17,241 个具有 validation informative rows 的 genes 也各为一个 batch；29 个真实极值/分位 forward+NLL+backward batches 的最大 allocation 为 10,081,855,488 bytes，未超过 shape estimator，投影为 3.198 hours/epoch，其中 2.674/2.676 raw host-I/O hours预计被当前 GPU 计算覆盖。该 profile 没有构造 optimizer、没有 `optimizer.step`、没有 gradient clipping、没有 checkpoint、没有完整真实 epoch，也没有 test prediction/metric；因此 3.198 hours 是不含 17,600 次 AdamW update 与 clipping 开销的 forward/backward/validation 投影，不得冒充真实端到端 epoch wall time。learning rate 与 validation-driven scheduler 不改变已 profile 的 batch shape；optimizer-update wall time 只能由获准训练的实际日志补充。当前 `configs/fabric_v2_full_cohort.yaml` 的资源已冻结但 `training_authorized=false`，状态只是 `READY_AWAITING_TRAINING_AUTHORIZATION`；本合同修改本身不启动训练。

## 16. Evaluation and scientific gates

### 16.1 Predictive evaluation

主要预测指标为原始 test distribution 上的 molecule-weighted compatible-path NLL，并辅以 compatible-set probability calibration 和 high-/low-DTU 分层结果。为审计内部剪接的长度与结构复杂度偏置，NLL 还必须按预冻结的 gene/path-token-length、\(D_g^{path}\)、\(V_g\) 与 legal-path-count strata 分层，并联结 §9.1 `PathScaleAudit` 的 \(\zeta\)/preactivation/logit/gradient/calibration 量；在局部 contrast 可辨识且存在独立 truth 时，internal choices 另按 relevant path token count 与 variable-edge/total-edge ratio 分层，不能由 TSS/PAS、短 path 或低-complexity genes 指标掩盖。为审计 ATAC atlas sampling density，DNA/Full 增益、mapping-valid coverage 与可报告 attribution coverage 还必须按预冻结的 cell-state abundance 及 ATAC neighbor-support metadata 分层；这些 metadata 不参与预测或样本加权。RNA event coverage 另按 §5.4 的 `RNAWindowCoverageAudit` waterfall 与 factor/region strata 报告，不把深内含子排除隐藏在总体 active-event 数中。长读长 observation coverage 另按 §3.3 `LongReadCompatibilityAudit` 报告，不把 matrix-catalog-incompatible exclusion 隐藏在 technical QC 或 NLL denominator 中。局部机制适用范围另按 §11.2 的 matched-context coverage waterfall 报告；它与 explanation-manifest coverage、ATAC-supported attribution coverage 互不替代。RETAINED_INTRON 相关预测与机制 summary 同时按 §3.4 的 path supervision-support tier 与 §3.3 的正交 `IR_biogenesis_context` 分层；主要表述限于 `processed-context RI-compatible path`，`mature_vs_nascent_unresolved` 不进入该 summary，除非另有 protocol-specific mature-transcript evidence 才允许 mature-RI 措辞。path/equivalence-group usage calibration、top-group accuracy 或 distribution correlation 只有在存在独立 truth，或预先冻结且报告不确定性的 EC deconvolution target 时才能作为附加指标；不能把模型自身分配当成 truth。只有满足 transcript-level supervision identifiability 的 singleton paths 才进入transcript-specific 附加评价。用户要求的 `full`/`atac`/`rbp` 模态比较必须在完全相同的 cells、genes、compatible rows、molecule weights 与 legal paths 上比较。

§15.4 的 `same_library_cross_pipeline_ont_matrix_agreement` 是上述 independent-truth 限制之外单独命名的 descriptive diagnostic：它只报告精确 log-space `ont_matrix_kl_count_weighted`，比较 held-out ONT matrix counts 与 compatible-read 模型在相同 library 上的 path probabilities，不能升级为真实 cellular isoform abundance accuracy、外部复现或 transcript assignment correctness。多个独立命令只有在 cell-gene IDs、transcript/path axes、ONT count weights、exclusions 与 denominator 完全相同时才可比较；当前 test 的既有 matrix-materialization exposure 必须随结果披露。

overall、high-DTU 与 non-high-DTU matrix agreement 分开报告。为避免引入未定义的 matching/reweighting analyst choice，第一版只做 train-frozen `support_stratified_sensitivity`，不产生单一“support-adjusted DTU difference”：分别按 `(a)` matrix path count 的固定 bins `2 | 3 | 4–5 | 6–10 | 11–20 | >20`、`(b)` gene-level train ONT raw count 的 \(\lfloor\log_2 x\rfloor\) 和 `(c)` gene-level train positive-cell support 的 \(\lfloor\log_2 x\rfloor\) 生成三张一维表；这里 \(x>0\)，bin assignment 只读取 train gene metadata，并在任何 model prediction 前冻结。每个 bin 内并列报告 high-DTU、non-high-DTU 的 cell-gene macro、count-weighted metric、gene 数、eligible cell-gene 数与 raw count denominator；不跨 bins 抽样、匹配、加权或汇总成调整后效应。任一 subgroup 分母为零时标记 `not_estimable`，不得合并/扩大 bin、补采样 held-out cases 或用 held-out performance 重选分层变量。该诊断不改变 primary full-cohort NLL、任何 test denominator 或 §14 的 DTU provenance 限制。

### 16.2 Mechanism-consistency evaluation

模型必须执行以下高信息量检查：同一 TSS event 对不同 downstream exon/PAS paths 能产生不同效应；内部 exon–exon 和 exon–PAS path-context dependence 可在合成数据中恢复；evidence-neutralization effect 在随机种子间稳定；预声明 effect 统计量在 motif-location shift、held-out pairing permutation、matched accessibility null 与 anchor-distance null 下是否衰减；global attention 是否把大多数 local events 变成近似 gene-wide uniform shifts。前两项与 cross-cell isolation 是实现 admission tests；其余是必须完整报告的科学 diagnostics，并按下述规则约束 claim，而不是靠模糊的“看起来减弱”决定能否运行训练。

event-density diagnostics 还必须按 §5.1 的 catalog/model-input token burden strata 分层报告预测性能和 attribution 稳定性，并报告动态 block norm 与 pre/post-normalization token norm。gate evidence-separation 则按 §6.3 报告 pairwise status、correlated-evidence set size、受影响 event/factor/channel 数量和联合 neutralization；它不能被 seed stability 汇总替代。attention saturation 与输出/梯度有限性在 §17 的合成 fixture 中验证，不要求正式评估侵入 `MultiheadAttention` 导出每个 token 的 raw attention logits。当前 V2 不再增加第二个 per-token cap、按 event count 缩放、event dropout 或第五个 burden ablation；若高负担 strata 仍出现系统性预测或 attribution 失效，应限制对应 claim 并另行修订 catalog/routing，而不是依据 test effect 临时改变 cap。

`ModelInjectionEquivalenceIndex` 是 Full-design 的冻结 reporting partition，必须报告 singleton/multi-member group 数、member-event 覆盖、footprint/motif-family relation、`should_have_collapsed_error` 和 selector 的完整/partial-class coverage。multi-member class 只有 joint-set effect 可以进入 primary sign agreement、\(D_Q\)、effect-rank、between-state 和 \(T_{attr}\) summary；exchangeable-member 与 partial-class diagnostics 不进入这些分母。某 modality-disabled ablation 中，该 channel 的 group attribution 固定为 `not_applicable_modality_absent`，不能把 Full design 中存在的 class 当作该 ablation 的有效注入效应。若本应 physical-collapse 的重复 rows 存在，属于 catalog implementation failure，不得用联合 attribution 掩盖。

`RouteDegreeCapAuditManifest` 必须在首次运行 degree/cap synthetic audit 前冻结 generator、planted targets、audit seeds、\(\epsilon_{syn}\)、recovery tolerance、seed-dispersion tolerance、degree-paired gradient-drift tolerance、cap-paired output/gradient-drift tolerance 和状态判定。balanced synthetic corpus 由同一个 shared model 同时拟合全部 conditions，不能为每个 degree 单独训练后靠参数重标定：(a) single-anchor family 固定 event/gate/features、两条 focal matched-context paths、downstream skeleton 和同一非零 planted \(\Delta\rho^\star\)，分别使用一 route 对一 distinct edge 的 \(d=2,4,8\)，新增 routes 只进入对称 nuisance branches且不执行 cap；(b) multi-anchor family 固定同一 focal event、graph、8 条 candidate routes 和 planted truth；focal anchor 的两条 routes 恒定保留，其余六条分属一个 4-route anchor 和一个 2-route anchor。三个预声明 matched catalog conditions 只改变这些 non-focal buckets 中、最终 gate-inactive 的静态 competitor load，使 non-focal retained routes 为 `4+2 | 2 | 0`、总 \(D_j^{post}=8,4,2\)。这样 pre-GraphGPS 注入变化可精确分解为 non-focal route removal 与 surviving-route renormalization 两项；由于 GraphGPS/global attention 和 path readout 仍可传递 non-focal structure，downstream \(\Delta\rho\) 只称为 combined cap-coupling sensitivity，不声称是纯 renormalization effect。

每个 condition、cell、event 和 audit seed 至少记录 full-minus-neutralized 的 \(\delta a^M=Gv^M\) per-edge norm median/max、event-total Frobenius norm、route \(L_1\) mass、\(\delta y\)、\(\delta\widehat y\)、\(\delta H\)、matched-context \(\Delta_j\rho\)、\(|\partial\rho/\partial G_j|\) 和 event-to-background pre-LayerNorm norm ratio。single-anchor 同方向 fixture 的 per-edge \(1/d\) 与 event Frobenius \(1/\sqrt d\) 只作 exact implementation identity。capability 以

\[
E_{s,q}
=
\frac{\left|\widehat{\Delta\rho}_{s,q}-\Delta\rho_q^\star\right|}
{\max\left(\left|\Delta\rho_q^\star\right|,\epsilon_{syn}\right)}
\]

及预冻结的 seed、degree-gradient 与 cap-paired tolerances 判断。状态不使用会掩盖双重失败的单值 enum，而保存独立字段：`implementation_valid`、`baseline_capability_pass`、`route_degree_pass`、`cap_coupling_pass`、`route_degree_catalog_applicable` 与 `cap_coupling_catalog_applicable`，再派生完整 reason list。结构恒等式、neutralization identity、有限性或复现失败令 `implementation_valid=false`；\(d=2\) 已无法恢复令 `baseline_capability_pass=false`；baseline 通过而 \(d=4/8\) 的 recovery 或 degree-paired gate-gradient drift 超阈值令 `route_degree_pass=false`；baseline 通过而 focal routes 不变的 external-cap variant 的 output 或 gate-gradient drift 超过 paired tolerance 令 `cap_coupling_pass=false`，两项可以同时失败。只有当前 `model_input` view 中不存在 \(d>2\) event 时，degree applicability 才为 false；只有不存在 `external_only_coupling` event 时，cap applicability 才为 false。catalog identity 变化后必须重审。

在 held-out test model inference 首次运行前，`implementation_valid=false` 阻止 implementation admission；`baseline_capability_pass=false` 必然阻止当前 routing 的 training admission；degree/cap applicability 为 true 时，相应 `route_degree_pass=false` 或 `cap_coupling_pass=false` 也阻止当前 routing 的 training admission。audit 不授权替代方程；后续候选只能经单独合同修订，在相同 synthetic、train/validation、seeds 与预算下选择唯一 runtime，并从头训练全部受影响的 commands。held-out test model inference 首次揭示的 failure 不得反馈 routing 选择，只按实际范围撤回 high-degree/multi-anchor/cap-coupled claims。真实 cohort 的 event-level gradient 与 attribution stability 必须按 event 自身的 exact \(D_j^{pre/post}\)、最大 per-anchor distinct-edge degree、固定 degree bins `1 | 2 | 3–4 | 5–8 | >8`、single/multi-anchor、cap status 和 \(\kappa_j^{renorm}\) 分层。gene-cell-level NLL 与 calibration 不得因同一 instance 含多个 events 而复制进多个 event rows；它们只按 manifest 预冻结的 gene-level summaries 分层：active events 的最大 degree、最大 \(\kappa^{renorm}\)、high-degree event fraction 及是否存在 `external_only_coupling`，并同时报告 model-active/reportable coverage。真实 degree-effect trend 因结构与生物学混杂，不能单独触发架构重选。

训练 runtime 不固定重复数。每条命令只记录一个 seed；用户需要评估优化稳定性时，可用相同 condition 和配置分别运行多个 seed，再在分析层显式汇总。任何跨 seed 离散度只衡量优化初始化稳定性，不是生物学 replicate、sampling uncertainty 或置信区间；不得依据 validation/test 性能删选或替换已经纳入汇总的 seed。

若用户为同一 condition 和配置运行多个 seeds，attribution 分析在同一个 explanation manifest 上逐 record 保存各 seed 的 \(\Delta\rho\)、\(\Delta\log P\) 与 \(\Delta P\)。主要数值可以是 across-seed median，同时报告 IQR 和

\[
\operatorname{sign\_agreement}(Q)
=
\frac{1}{S}
\max\left\{
\sum_s\mathbb 1[\Delta_sQ>\epsilon_{num}],
\sum_s\mathbb 1[\Delta_sQ<-\epsilon_{num}]
\right\},
\]

其中 \(\epsilon_{num}\) 是在 synthetic/reproducibility tests 中预冻结的纯数值容差，不能由 test effect 大小选择；落入容差的 seed 计为不一致。只有 `sign_agreement=1` 的 record 才能进入有方向的 event-mechanism summary，否则标为 `seed_unstable` 并只作诊断。还必须按 interaction-support strata 报告 seeds 间 effect ranking 的 pairwise Spearman correlation。原始 \(W^{int}\) coefficients 不要求跨 seed 一致，也不得作为替代 estimand。

效应幅度另定义：

\[
D_Q
=
\frac{
\operatorname{IQR}_s(\Delta_sQ)
}{
\max\left(
\left|\operatorname{median}_s\Delta_sQ\right|,
\epsilon_{effect,Q}
\right)
}.
\]

\(\epsilon_{effect,Q}\) 是在 validation explanation manifest 上、按输出类型预冻结的最小可报告效应尺度，不是浮点容差 \(\epsilon_{num}\)，也不能看 test 后修改；\(D_Q\) 的允许上限在同一 validation 阶段冻结。只有 \(\left|\operatorname{median}_s\Delta_sQ\right|\ge\epsilon_{effect,Q}\) 且 \(D_Q\) 不超过该上限时，才允许声称稳定贡献幅度。否则标为 `below_effect_reporting_floor` 或 `magnitude_seed_unstable`；若 sign gate 通过，仍可报告方向和完整 per-seed range，但不得把 across-seed median 称为稳定贡献幅度。正式 attribution 同时报告这些互不替代的稳定性状态。

状态内 attribution 与状态间差异是不同 estimands。对预先冻结的 selector \(S\)、输出 \(Q\)、reporting state \(z\) 和 seed \(s\)，令 \(\mathcal I_z(S,Q)\) 只含该状态中 `supported_context`、非 extrapolation、且满足目标 local/path supervision 与所声称 evidence-separation scope 的 cells；同一 cell 对同一 record 只计一次。该 cell-ID set 必须在读取任何 checkpoint effect 前，由输入 masks、support、identifiability 与冻结 collinearity scope 构建一次并供纳入该分析的全部 seeds 共用；不得按 per-seed sign、magnitude、\(D_Q\)、预测正确性或输出是否有利筛 cells。相关 evidence 的 joint-set state contrast 可以进入该集合；若要比较单一 DNA-vs-RNA 或 factor-A-vs-factor-B 来源，则 §6.3 的来源分离条件必须通过，否则只能标记为 correlated-evidence model contrast。定义典型细胞状态效应为等 cell 权重中位数：

\[
\theta_s(S,Q;z)
=
\operatorname{median}_{i\in\mathcal I_z(S,Q)}
\Delta^{(s)}_{S,i}Q,
\]

以及预声明状态对 \(z_1:z_2\) 的差异

\[
C_s(S,Q;z_1:z_2)
=
\theta_s(S,Q;z_1)-\theta_s(S,Q;z_2).
\]

等 cell 权重使其回答“典型被观测细胞”的模型反事实差异，不把长读长测序深度变成状态权重。若需要 donor/embryo population claim，必须另有 donor-aware aggregation 或相应 held-out split；优化 seeds 不能代替生物学 replicates。state definitions、state pair、selector、choice/path/group、matched-context signature 以及每个状态的最小 eligible unique-cell 数 \(n^{state}_{min}\) 必须在 test 前冻结，并要求 \(|\mathcal I_z(S,Q)|\ge n^{state}_{min}\)，而不是用原始 state cell count；任一状态不足时记为 `state_contrast_not_estimable`，不得扩大状态或合并 donors 补足样本。

在 validation state-contrast manifest 上按输出类型冻结 \(\epsilon^{state}_{effect,Q}\) 与 \(D^{state}_{Q,max}\)。任何正式“状态 \(z_1\) 高于/低于状态 \(z_2\)”claim 都要求三个 \(C_s\) 全部在同一方向且超出 \(\epsilon_{num}\)，并要求最小状态差异

\[
\left|\operatorname{median}_s C_s\right|
\ge
\epsilon^{state}_{effect,Q}.
\]

通过方向与 effect-floor gate 后，若还要引用一个稳定的状态差异数值幅度，进一步要求

\[
D^{state}_Q
=
\frac{
\operatorname{IQR}_s(C_s)
}{
\max\left(
\left|\operatorname{median}_s C_s\right|,
\epsilon^{state}_{effect,Q}
\right)
}
\le
D^{state}_{Q,max}.
\]

两个 states 的 record-level supervision、pairing、state-residual 与相关 evidence claim restrictions 必须同时通过。未通过 direction 或 effect-floor gate 时只能逐 seed 作诊断报告，不能作状态强弱 claim；通过这两门后才可表述“在当前 observed cohort 的 FABRIC model counterfactual 中，状态 \(z_1\) 的贡献高于/低于状态 \(z_2\)”，再通过 \(D^{state}_Q\) 才能把跨-seed median 称为稳定差异幅度。一个状态自身低于单状态 effect floor 不能被表述为“无效”，只能称低于预声明的可报告效应尺度。真正的近零等价 claim 需要另行冻结 equivalence margin，当前 V2 不提供。

held-out pairing permutation 是固定 checkpoint 下的 inference-only diagnostic，不产生因果或随机化检验 \(p\)-value。模型参数、source normalization 与 train gate baselines/scales 完全冻结。primary coarse null 使用 `stage × developmental system × donor` strata；secondary strict null 在样本量允许时增加 `cell type`。每个 stratum 至少 20 个 cells，否则该 stratum 记为 `not estimable`，不得退化到更宽 strata。

factor/RBP null 对整个 raw activity-entity row vector 及其 observation masks 使用同一个 cell permutation，因此 factor-factor 相关结构、group definitions 和一个 entity 在所有 genes/events 中的共享关系保持不变；置换后重算 gates。ATAC null 使用另一个共同 cell permutation，联合移动完整 mapped peak vector、\(m_i^A\) 与只作诊断的 mapping-support metadata，同时保持 factor activity 不变。不得逐 event 独立洗牌，不得直接洗牌 standardized gates，也不重新训练。

每种 null 固定运行 100 次，并对全部纳入该分析的 seeds 使用完全相同的第 \(b\) 次 permutation assignment。每个 seed 先计算完整冻结 test cohort 上的 \(T^{(s)}_{NLL,b}=NLL^{(s)}_{perm,b}-NLL^{(s)}_{paired}\) 和 explanation manifest 上所有可报告 contrasts 的 \(T^{(s)}_{attr,b}=\operatorname{median}|\Delta_S\rho|\)，然后以 seeds 间 median 得到唯一的 \(T_{NLL,b}\) 与 \(T_{attr,b}\)；paired \(T_{attr}\) 同样先逐 seed 计算再取 median。95/100 gate 只对这组预冻结 aggregate statistics 应用，不允许挑选单个 checkpoint。分别报告 paired 值与 100 次 null 的 median/IQR。没有 contrasts、paired \(T_{attr}=0\) 或缺少合法 strata 时报告 `not estimable`，不选择性删除零效应，也不事后改统计量。coarse 与 strict null 分开报告；二者检查真实 cell–evidence pairing 与细粒度 cell-state proxy 的不同风险，不能互相替代。

若要作“模型使用了真实 cell–evidence pairing”的正式 claim，coarse null 必须同时满足：100 次中至少 95 次 \(T_{NLL}>0\)，且 paired \(T_{attr}\) 严格高于 100 次 null \(T_{attr}\) 的 empirical 95th percentile。若 claim 进一步限定为 cell-type 内效应，secondary strict null 也必须满足同一规则；若 strict null 为 `not estimable`，只能保留 coarse-state 范围的表述。该 95/100 规则是预冻结的 descriptive claim-admission criterion，不报告为随机化检验 \(p\)-value。未通过不阻止报告预测性能或启动已获授权的训练，但必须撤回相应 pairing-dependent mechanism claim。

motif-location shift 必须保持 factor/group、motif-score bin、region type 与候选可路由性；matched-accessibility null 必须保持 peak accessibility/support 与 cell-state strata；anchor-distance null 必须保持 modality、region、geometry kind、site-window signed-distance bin 或 edge-contained relative-position/boundary-distance bins，以及 event burden。具体 bins 和 matching tolerance 在 validation 前冻结，不能看见 test effect 后寻找更有利的匹配。上述 null 检查固定模型是否依赖真实 cell–evidence pairing 与局部几何，不能替代 retrained predictive ablation，也不是因果检验。

### 16.3 Biological validation

生物学验证包含两条独立证据轴。第一条是 perturbation：对 factor activity、RBP perturbation 或 accessibility perturbation 重算 gates，并比较预测的局部/路径变化与实测变化。第二条是具体 case：例如 CTCF–Pol II–CD45 exon retention、alternative promoter/TSS、alternative PAS，以及有明确 motif、状态和 isoform readout 的 RBP cases。

case study 只能证明与已知机制一致；只有适当干预设计才能支持因果语言。ATAC 图谱性质与质量证据应在论文前半部分独立建立，FABRIC 在后半部分连接 ATAC、局部 RNA processing 和完整 isoform paths。

### 16.4 State-residual gate

对每个 `cohort_contrast_separable` group \(E\) 和具有正 informative molecule mass \(N_{i,g}=\sum_{k\in\mathcal K^{inf}_{i,g}}n_k\) 的 held-out \((i,g)\)，定义 compatible-likelihood score residual：

\[
r_{i,g,E}
=
\frac{1}{N_{i,g}}
\sum_{k\in\mathcal K^{inf}_{i,g}} n_k
\left[
\frac{\sum_{p\in C_k\cap E}P_{i,g}(p)}
{\sum_{p\in C_k}P_{i,g}(p)}
-P_{i,g}(E)
\right].
\]

分母严格为正是该 `(cell,gene)` 进入 likelihood-informative evaluation universe 的前置条件；不满足者不进入此 residual，不能在此使用 epsilon 修补或重新分类 `LongReadCompatibilityAudit`。该 residual 是观测 compatible rows 相对于模型 path distribution 的 score-like discrepancy，不是无歧义的单细胞“实测 DTU”。

state latent 固定为只用短读长总表达、且不读取 long-read path/DTU target 或 ATAC 的 train-only PCA scores；HVG 规则、维数、中心化与投影在 test 前冻结。用 validation residuals 拟合两个 molecule-weighted ridge diagnostics，并在 test 上冻结评价：nuisance model 只含 gene/group intercept、\(\log(1+N_{i,g})\)、正 EC-row count 和 donor；state model 在完全相同项上增加 stage、developmental system、cell type 与 frozen RNA state latent。ridge penalty 只用 validation cross-validation 选择，不能读取 test。

主要 gate statistic 是 test 上的增量加权解释度：

\[
\Delta R^2_{state}
=
R^2_{state}-R^2_{nuisance},
\]

权重为 \(N_{i,g}\)，overall eligible cohort 与预声明 high-DTU stratum 分别计算。若 Full 模型任一层面的 \(\Delta R^2_{state}>0.05\)，named-event catalog 仍留下不可接受的系统性 state residual：V2 可以继续报告预测性能和这一失败结果，但不得作正式的 cell-state-specific event mechanism claim。\(0.05\) 是本合同在查看 test 前冻结的 admission tolerance，不是普适生物学常数；若要修改必须单独修订合同并重新冻结 test。不得通过 test-informed State branch、metadata embedding、新 loss 或 residual model 回灌主模型来绕过该 gate。

## 17. Required tests

V2 实现至少需要以下高信息量测试：

1. gene graph、ordered path edges 与 GTF transcript identity 完全一致，`local_edge_index` 只含合法路径上的双向 consecutive-edge pairs，union graph 不生成新 paths；
2. 同一 `(gene, cell)` 的多个 EC rows 复用同一 path logits，跨细胞 attention 严格隔离；
3. 物理 event identity、factor/group identity、coordinates、strand 与 routing 对齐；connected-component collapse 对链式重叠结果唯一；同一 genomic hit 被重叠 windows 扫到时形成一个 event 和多个 routes；预声明为等价且重叠的同 factor PWM hits 精确折叠并保留全部 source motif provenance，非等价 motifs 不误合并；cap bucket 包含 `cap_evidence_class`，motif-anchored DNA 与 accessibility-only 不互相驱逐，而 unique 与 factor-equivalence-group events 仍在同一 motif bucket 中按同一排序竞争；
4. ambiguous motif 只能输出 factor-group claim，group raw counts 必须在 CP10K-log1p 前求和；绝对坐标、event/peak/motif/family/gene/transcript IDs 不进入 \(u_r^{base}\) 或 \(u_r^{int}\)；
5. 0-based half-open 坐标、canonical center、正负链 oriented distances、`OVERLAP_ANCHOR`、site-window 与 edge-contained geometry 按 §7 分别计算，NA/mask 不被零值替代；pre/post-cap route identity、route-record 数与 distinct-edge degree 不混淆，production route weights 对每个 retained event 求和为一，\(D_j^{post}=0\) 的 event 不得 model-active，任一 evidence set neutralization 删除其全部且仅其 routed terms；
6. canonical supported-rectangle interaction design 的 raw-cell/route order、exact-rank pivot selection、zero padding 和 active mask 可复现，完整 \([u^{base},u^{int}]\) 无零列、exact duplicate 或未记录的 rank deficiency；对通过 raw-support与active-span admission 的不同 \(q_r\) 类别，`factor_A-downstream + factor_B-upstream` 与交换配对后的投影前 aggregate feature必须不同；关闭 \(u_r^{int}\) 的负对照应复现碰撞，并以固定 identity/selector projection验证该差异可被 \(W_D^{int}/W_R^{int}\)表达，但不要求任意训练后的模型输出必然不同；
7. RNA/ATAC 均为 CP10K-log1p，DNA 必须 product-before-z-score；`mapping_valid` 唯一决定 \(m_i^A\)，ATAC ESS/evenness/coverage/weight-concentration metadata 不单独或联合参与 `mapping_valid`，也不进入 gate、\(\mu/\sigma\) 权重、logits、loss 或数值 attribution；固定 mapped \(A\) 与绝对 mapping QC 时，改变这些 metadata 不得改变 \(m_i^A\)、模型输出或数值 attribution；
8. measured zero、missing observation、invalid mapping 和 low neighbor support 四种语义不混淆；一个近邻与该邻居的多个完全相同副本必须产生相同 mapped \(A\)、gate 和 logits，uniform-but-distant neighbors 即使 ESS 较高仍须由绝对 mapping QC 判定；
9. gate-key mean/scale/eligibility 只由 train split 拟合并在 validation/test 冻结；有效 cells、gate-level effective cell count、informative molecules 和 dynamic SD 门槛逐项执行，且不与 ATAC-neighbor ESS 混淆；inactive events 留在 audit catalog 但不进入模型 tensors；support quantiles 使用相同 \(\omega\) population，held-out/source-proxy/member-count perturbation 的 raw \(b\)、standardized residual \(z\)、final gate \(G\) 与 out-of-support flags 正确；
10. static CIS normalization 在 unique structural edges 上每 edge 一次、按 availability mask 拟合，不能随 gene-cell duplication 或 molecule coverage 改变；
11. self-factor flag 依据 `activity_gene_ids` 与 target gene 生成，group-member perturbation 重算完整 group proxy，且 activity proxy 不读取 path usage/DTU label；
12. 三个 CLI conditions（`full`、`atac`、`rbp`）保持相同输入宽度、tensor shapes、总参数量与 seed-controlled 初始化；每条命令只训练所选 condition，未使用模态在 GraphGPS 前正确置零，不自动训练 CIS-only、architecture comparator 或其他 seed；
13. path pooling 的 gene-centered residual sum、first、last 和 `log_edge_count` 按去重后的合法 structural paths 与转录方向正确；固定 contextual states 时，所有 paths 共有的 constitutive edge 对任意 path-pair residual difference 严格抵消，增加共同 constitutive tokens 不改变 \(\zeta\) difference 及其对 differential tokens 的 Jacobian，mean-only 负对照复现 \(1/|p|\) 衰减；完整模型另按 path length 诊断 relative log-mass 与 internal-choice performance，不要求严格 padding-invariant；不同 supervised paths 执行 representation-collision audit；
14. compatible-path NLL 与 brute-force reference 在 toy 和真实 fixture 上一致；
15. observational-equivalence grouping、rank-deficient distinct columns、augmented-rank、full-operator row-space estimability、held-out split-group policy、exclusive support 和 support tiers 正确；group probability 精确等于成员 paths 概率之和，跨 alternative/eligible/context boundary 的 group 会禁止 local attribution；
16. marginal 与 matched-context signatures/path subsets 正确，二者 relative log-mass 对 full/counterfactual 各自独立的 gene-wide logit shift 均不变，centered logit change 使用 unique structural paths 等权 gauge；
17. 单 event、同 factor/group 和同 anchor-region 三类 primitive evidence sets 的删除范围准确；correlated-evidence set 只取这些 primitive sets 的确定性 route union，不引入第四种可学习 selector；
18. baseline neutralization、source-proxy perturbation、member-count perturbation 与独立 `observed_library_context` 产生符合各自定义的不同 scope/value；缺失所需动态 context 不得以零效应进入 supported attribution；
19. explanation manifest 不读取 test outcomes，并正确报告选择覆盖率；
20. inference-only permutation null 不改变 checkpoint 或 train baselines/scales，只在合法 strata 内联合置换 raw evidence/masks，并复现冻结的 \(T_{NLL}\) 与 \(T_{attr}\)；
21. 合成 gene 中同一 TSS event 能对两个不同 downstream paths 产生不同符号或幅度的 effect；
22. 所有主模型参数获得有限梯度，输入和输出无未声明 NaN/Inf；任一未声明非有限值使对应 run/stratum evaluation invalid 并禁止其 prediction/mechanism claim，不得仅降级为 magnitude instability；
23. state-residual 的 score residual、validation-fitted diagnostics 与 test \(\Delta R^2_{state}\) 按冻结公式复现，且不回灌主模型；
24. 相同固定输入和 seed 下关键 logits、probabilities、group sums 与 attribution 在合理容差内可复现；
25. `InteractionSupportManifest`只由train split生成并冻结；支持网格 `A:{0,1}, B:{0,1,2}, C:{1,2}` 必须复现旧 treatment-column mask在`A/0`与`C/2` reference下产生不同 admitted spans的回归见证，而canonical supported-rectangle basis在两种base recoding下满足相同 raw-cell span、raw contrast/comparator status和claim set，子空间以exact rank而非column名称比较；full support恢复 \((|F|-1)(|L|-1)\) rank，无supported rectangle得到rank 0，focal两arms但无comparator得到`within_factor_only`，comparator四角完整但contrast不在active span时不得作factor-specific grammar claim；构造两个context fields在route rows上exact/general-linearly aliased的fixture，交换manifest field order不得把field-specific claim从一方转给另一方，两者均标记`cross_field_context_not_separable`且只允许joint-field diagnostic；每个无comparator的q-summary以`comparator_id=NULL`承载状态，每个event的多个field/level-pair contrasts保持独立records；validation/test不能激活新basis columns；关闭unsupported/rank-redundant interaction directions后physical event、gate、factor baseline与共享route-context baseline仍保留，不声称后续网络不存在任何隐式interaction；
26. optimizer 参数组完整且互斥，bias/norm 不 decay，\(W_D^{int}/W_R^{int}\) 使用相同的 \(\lambda_{int}>\lambda_{base}\)；在 likelihood gradient 为零的最小 fixture 上一步更新产生预期的 decoupled decay，且不误改不应 decay 的参数；`constant|reduce_on_plateau` 条件校验、逐命令 override、validation-NLL-only scheduler step、LR history 与 checkpoint optimizer/scheduler state 可复现；
27. null rare-interaction synthetic fixture 在一组显式 seeds 下被 support gate 或 interaction shrinkage 抑制，充分支持的 planted interaction 仍可恢复；若用户对同一运行身份汇总多个 seeds，真实 attribution 才报告 median、IQR、sign agreement、\(D_Q\) 与 effect-rank Spearman correlation，并按预声明规则标记方向或幅度不稳定；
28. §10 与 §15.2 的 train loss 对同一 inputs 数值一致，validation/test 使用各自固定 molecule denominator；每个 train-positive gene 每 epoch 恰好一次 optimizer step，gene 内 adaptive microbatches 只累积梯度，global-norm clipping 只在 gene 梯度完整后、step 前执行一次，step loss 的 \(G\,N_g/n_g/M_{train}\) 缩放可复现，改变 shape packing 不改变 update；同 edge count 但 route/path/EC shape 不同的 genes 得到相应不同的 memory plan，超过 CUDA kernel cell cap 时确定性拆批；每个 `TrainingRunManifest` 的 optimizer/training controls 与 resolved config 一致，多个独立命令只有在 learning-rate policy、clipping、penalties、cap、split 和其他比较身份相同时才能作严格 paired comparison；
29. 固定观测 cell 与模型参数时，single-event neutralization 在 modality aggregate \(a^M\) 处、即 \(W_X\) 和 pre-LayerNorm 之前的 full-minus-neutralized 差分严格等于 \(G^{obs}v^M_{j,e}\)；只改变一个 shared gate key 时，aggregate 差分严格等于 \(\Delta G\sum_{j:\tau(j)=\tau}v^M_{j,e}\)，且 joint projected 差分等于 \(W_X\) 对相应嵌入差分的线性像；pre-LayerNorm 后及完整模型输出不被测试强制为仿射、线性或单调；source-proxy/member-count response sweep 按声明的输入层级重建全部受影响 gates/events，并正确标记 train-support 与 extrapolation；
30. physical-event collapse 与 event 内 route-weight normalization 不复制 event mass；cap 明确按 `(anchor group, cap_evidence_class)` 而不是按整个 anchor 或 token 执行，DNA 两类 bucket 均饱和时该 anchor 可有 32 个 retained events；`RouteDegreeCapAudit` 精确复现 cap loss、renorm gain、anchor-mass decomposition、single/multi-anchor transitions 与 `external_only_coupling`；相同 token 上 1、4、16 个同 gate、同方向 events 的 synthetic fixture 中，hidden width 为 \(H\) 的固定无 affine pre-LayerNorm 必须满足 \(\lVert\widehat y_{i,e}\rVert_2\le\sqrt H\)，且 attention 输出、模型输出和梯度有限；test-local reference calculation 删除 pre-normalization 后应复现显著的 pre-attention norm 与 attention-score 放大，但不得为此在生产模型增加可切换分支，也不强制特定的精确 \(N\) 次幂；相同 event 数但 shared/different gate keys 得到预期不同的 \(B^{gate}\)，catalog burden 与仅含 `model_active` routes 的 model-input burden 不混淆，全部 audit fields 与分层报告可复现；
31. `RNAWindowCoverageAudit` 的 eligible reference denominator provenance、基于全部 legal transcripts 的区域分类优先级、window membership、legal-route、cap-retained 与 train-derived model-active waterfall 可复现；前四步与 active suffix 的 split 语义不混淆，外部实验位点覆盖与 motif-candidate coverage 不混名；
32. `RETAINED_INTRON` evidence fixture 区分双 boundary、单 boundary、intron-only、excising junction、multi-intron unspliced、`IR_evidence_censored`、processed-context support、internal priming 与 genomic-DNA flags；不充分 IR evidence 删除后只用剩余合格证据重建 \(C_k\)，无剩余区分证据时得到 \(C_k=\mathcal Y_g\)；降解/截短造成 compatible-set 变宽而不自动产生 IR-positive evidence，`IR_biogenesis_context` 与 `PathIdentifiabilityIndex` 正交，library/donor QC 汇总可复现；
33. `TrainingRunManifest` 恰好包含一个命令 seed 和一个 `full|atac|rbp` condition，并冻结 resolved learning rate、scheduler 及参数、gradient clip norm、penalties、max epochs、early-stopping patience、per-gene cap、`optimizer_step_unit=train_positive_gene`、gene-microbatch accumulation、adaptive GPU target、kernel cell cap 与 backed-shard prefetch identity；拒绝 embedded seed lists、condition lists、独立 `sampling_seed` 与内部 condition 名；相同 seed/condition/config/input 重建相同初始化、逐 epoch gene-cell 抽样和 gene update order，重复 seed 或 condition 由用户分别提交命令；
- completed-epoch recovery fixture 在相同 seed/condition/resolved config/input 上验证 uninterrupted 与 interrupt-after-atomic-`latest.pt` 后 `--resume-from` 的逐 epoch history、monitor、best epoch/NLL、最终 model、optimizer、scheduler 与 RNG continuation 完全一致；篡改 seed、condition、config、有序 gene identity、checkpoint epoch/history/early-stopping state 或改用 `best.pt` 必须明确失败，同一 run-dir 的并发 writer 必须被拒绝；恢复不产生额外 validation traversal，不读取 test，也不重复已完成 epoch；
34. `LongReadCompatibilityAudit` 仅以 model-admitted、具有非空冻结 \(\mathcal Y_g\)、pre-compatibility technical-QC-pass 且正 molecule mass 的 rows 为统计总体；将其互斥完备地分成 \(C_k=\varnothing\)、\(\varnothing\ne C_k\subsetneq\mathcal Y_g\) 与 \(C_k=\mathcal Y_g\)，逐 stratum 验证三类 mass 之和等于总体 mass，并复现分开的 count/mass/fraction；零分母标为 `not_estimable`，technical failure 与 `no_matrix_isoform_compatible` reason codes 不混淆，后者不进入 likelihood 也不被自动标成 novel isoform；
35. 相同 focal internal swap 被嵌入递增 \(D_g^{path}\)/\(V_g\) 的合成 catalogs 时，未缩放 residual sum 仍保持共同 constitutive padding invariance 与局部差分等式，并可复现 `PathScaleAudit` 的 \(\zeta\)、preactivation、logit、gradient 和 calibration strata；unit-test-only 候选 \(1/\sqrt{\max(1,D_g^{path})}\) reference calculation 只验证等比例 pair difference 与共同 padding invariance，不成为 production runtime 开关；
36. `GateCollinearityAudit` 只使用共同有效 train cells 和冻结 molecule weights，正确区分完全/近似共线、负相关、共同方差为零、联合支持不足与未超过阈值；pairwise edges、connected reporting sets 与联合 neutralization 可复现，correlated set 不改变 event identity，单 seed 或多 seed 稳定的单来源 attribution 都不能绕过 evidence-separation claim restriction；
37. `AlternativeReportingIndex` 对所有 structurally valid choices 以冻结 `contrast_id` 去重 alternative pair，复现 matched-context candidate 与 cohort-reportable counts；choice-level waterfall 的最后一级要求同一 held-out record 同时 direct-cell-supported 且 manifest-selected，record-level coverage 使用每个 split 内冻结的 eligible cell-gene × two-arm candidate 分母，零分母标为 `not_estimable`；TSS/internal/PAS choice fractions 与 pair×context counts 不更换统计单位；
38. between-state fixture 使用等 cell 权重、预冻结 state pair 和两侧 eligible cells 计算每个已运行 seed 的 \(\theta_s\) 与 \(C_s\)，正确执行每状态最小细胞数、可选跨-seed方向规则、effect floor 和 \(D^{state}_Q\) gate；一个状态低于报告阈值不会被误标为零效应，优化 seeds 不被解释成生物学 replicates；
39. `InteractionSupportManifest`按`(channel, context field)`分别复现`N_raw_rectangles_potential`、`N_four_corner_supported`、`N_support_span`、`N_rank_retained`与固定`N_padded`，rectangle count、support-span rank和combined-design rank不混淆；`OPEN_ONLY`不进factor-specific分母；无supported rectangle标为`not_applicable_no_supported_rectangle`，其余`zero/partial/full` basis coverage与逐raw-contrast的`unsupported_focal_arms | within_factor_only | raw_contrast_not_in_active_span | cross_field_context_not_separable | factor_specific_grammar_estimable` claim scope一致。
40. `ModelInjectionEquivalenceIndex` 由最终 per-edge \((\beta,\iota)\)、modality 与 gate key 精确复现：不同 route-record 分解但相同 aggregate 进入同组，任一最终 signature aggregate 的精确差异则分组；masked-off interaction 或纯 provenance 差异不拆组；两个 correctly-distinct physical members 均保留且 full aggregate 精确含两份注入，分别 neutralize 得到相同 tensors/logits/probabilities，joint neutralization 删除两份且不强制等于 member effects 之和；非等价 motif families 不被 forward dedup，本应 physical-collapse 的残留重复明确失败；partial-class selector 只作诊断，完整 group union 才进入正式 summary，index 不读取 held-out outcome、checkpoint、seed 或 attribution magnitude；
41. balanced shared-model route fixture 同时复现 single-anchor \(d=2,4,8\) 与预声明 `4+2 | 2 | 0` non-focal-bucket competitor conditions 的 8-route multi-anchor cap family；per-edge \(1/d\)、event Frobenius \(1/\sqrt d\)、cap/renormalization decomposition、all named full-minus-neutralized intermediates、matched \(\Delta\rho\)、gate gradient、recovery error、预声明 audit seeds、六个独立 validity/pass/applicability fields、可同时出现的 degree/cap failures 及 training guard 均按冻结 `RouteDegreeCapAuditManifest` 重现；输入层比例本身不作为 capability failure，且 broadcast、degree correction、site carrier 或 runtime routing switch 不得出现在 production model 中；
42. ONT matrix-agreement fixture 先验证完整 matrix-row/cell/path identity；mapping 缺失、重复、多对一、model path 缺失、model 额外 path 或 validation cell/split drift 任一出现都必须 fail closed。动态 scope 只含互斥的 zero-total、少于两个阳性 matrix paths 与 eligible 三类；fixture 以 float64 log-softmax 精确复现 count-weighted \(D_{KL}(q\Vert p)\)，验证 cell-gene/count-mass denominator，并拒绝 non-finite logits；不计算 top-1、top-5、singleton、macro、CE 或 PRISM-clamped compatibility fields。
43. 每个 completed training epoch 后恰好一次、无 initialization/mid-epoch，以同一 frozen model state、`eval`/`no_grad` 和完整 validation traversal 计算 `validation_compatible_path_nll` 与 `ont_matrix_kl_count_weighted`；不得完整遍历 train，也不得为 ONT metric 触发第二次 validation forward。只有 NLL 参与 checkpoint selection；ONT KL log 保持 sealed 且 selection-ineligible；test model predictions/metrics 在 checkpoint 与规则冻结且 `final_test_authorized=true` 前不可计算。
44. real-data `OntObservationProcessAudit` 必须复现 matrix quantifier 与 compatible-read artifact 的软件/config/reference/GTF/feature/barcode/QC/assignment provenance，以及按 split/cell-gene 的 matrix count、pre-compatibility mass、empty/proper-subset/full-set compatible mass 和其他明确 fate 的 conservation/overlap。compatible-read rebuild 或该 audit 未完成时 monitor admission 必须为 `PENDING_OBSERVATION_PROCESS_AUDIT`；若两套 molecule populations 无法证明相同，schema 与报告标题稳定保留 `cross_pipeline`，不得降级成无提示的 `same_observation` accuracy；
45. `CompatibilityArtifactManifest` fixture 从冻结 alignment/path/split inputs 重现 molecule fate、ordered \(C_k\)、整数 EC mass 与逐 split conservation；17,706 个 structural candidates 全部有显式 support status，实际 likelihood genes 只由 train positive informative mass决定。缺 gene、identity drift、train/validation policy drift、非整数/不守恒 mass 或把历史 7,198-gene EC 冒充完整输入均拒绝 admission；若 test compatibility 未暴露，checkpoint 前 producer invocation 不读取/写出 test rows，若已暴露则 exposure marker 必须保留。

若以后启用 §4.1，另需验证 AlphaGenome checkpoint/reference/GTF/coordinate/flank/pooling identity、离线 embedding 与 edge-token 对齐、缓存复现性，以及 `explicit CIS`、`explicit CIS + AlphaGenome` 的独立 retrained ablation；这些测试不属于当前 V2 主模型的完成条件。

## 18. Claim boundary

对于下述 event-to-alternative-to-path 的 cell-state-specific mechanism 表述，V2只有在当前cell的必要dynamic observation masks有效、对应event/gate未被标记为`missing_context_not_estimable`或`model_extrapolation`，且§§3.4/11的supervision-identifiability、§16.2的跨seed方向稳定性与pairing claim gate、§16.4的state-residual claim gate对目标claim均通过时，才允许报告；若要声称稳定贡献幅度，还必须通过\(D_Q\) gate。若同时描述该 event 所在的 factor×context design scope，预声明 raw contrast 必须在§5.5中达到`factor_specific_grammar_estimable`并列出实际通过active-span与cross-field-separability gates的comparator IDs；这只允许说“该factor-specific positional contrast在设计中可辨识”，不能说模型已经学得相对comparator非零的位置偏好。当前V2没有冻结后一个更强命题所需的comparator-linked output DID。`within_factor_only`只能称within-factor model context contrast，`cross_field_context_not_separable`只能称joint-field diagnostic，`raw_contrast_not_in_active_span`或`unsupported_focal_arms`均不能把方向归为该factor特有的位置偏好。§6.3的`correlated_evidence`或`evidence_separation_not_estimable`不禁止报告单来源模型反事实，却禁止把它升级为DNA-vs-RNA或factor-A-vs-factor-B的唯一来源分解；应显式标记并优先报告correlated set联合效应。状态\(z_1\)与\(z_2\)的强弱比较还必须通过§16.2单独的between-state estimand、样本支持、三-seed方向和幅度门，不能由两个state-specific medians的肉眼差异替代。未通过上述gates时仍可报告明确降级为一般预测性能、模型反事实或诊断的结果，但不能使用下述细胞状态机制表述：

> 在 FABRIC 模型中，具名 factor/entity 的具名 DNA/RNA motif event 在某类细胞状态下，使 alternative \(a\) 相对于 \(b\) 的 matched-context 或 marginal relative log-mass 改变 \(\Delta\rho\)，并使监督可辨识的 unique matrix structural path（保留 transcript aliases）或 observational-equivalence group 的 matrix-catalog-compatible observed-library conditional log probability/probability 改变相应幅度。

上述“具名 motif event”措辞只适用于singleton `ModelInjectionEquivalenceIndex` class。multi-member class的主要表述必须改为：

> 在 FABRIC 模型中，该组模型注入不可区分但物理记录保持独立的 DNA/RNA motif-event annotations，在当前细胞背景下联合改变了指定 alternative contrast 与 path/group probability。

不得从该set-level claim升级为任一member motif family、binding mode或唯一位点的单独机制；即使members位于相同或重叠footprint也不例外。`exchangeable_member_counterfactual`与`partial_model_injection_group`只能作为诊断。

对于 accessibility-only event，唯一允许的平行表述是：

> 在 FABRIC 模型中，某个具名 anchor region 的 accessibility-only event 在某类细胞状态下，使指定 alternative contrast 与 matrix-catalog-compatible observed-library path/group probability 改变相应幅度。

此类结果的 `factor_entity_id=NULL`、`motif_id=NULL`；不能因为邻近存在 TF motif 就把 accessibility-only contribution 归给该 TF。

V2 不允许仅凭观察数据报告：motif hit 等于真实结合；ATAC openness 等于 TF occupancy；RNA expression 等于蛋白活性或 factor 浓度；预测曲线等于可独立解释的 Hill kinetics、occupancy、阈值或 cooperativity；global attention 学出了转录方向；TSS causally controls PAS；某个 factor event 具有湿实验因果效应；隐式神经网络交互代表 TF–RBP PPI；单事件 masking 是唯一机制分解；marginal contrast 是原生局部剪接能量；监督不可辨识组内的网络分配得到 transcript-specific 数据支持。motif 无法区分家族成员时，不得从 factor-group claim 升格到单 factor。

因此所有 model-derived results 默认属于 `motif_and_cell_context_supported_association` 或 `model_counterfactual_contribution`。只有独立 perturbation evidence 可以升级对应、明确限定的 causal claim。

## 19. Relationship to PRISM and Otari

### 19.1 PRISM

PRISM 已经提供 per-gene processing graph、GTF path catalog、gene-level softmax 和 compatible-path NLL。FABRIC V2 不把这些继承部分宣称为新贡献。PRISM 的动态 RBP/ATAC context 在静态 CIS GraphGPS 之后通过 edge-local cross-attention 加入，随后每条 edge 先标量化，path logit 是 edge energies 的线性和。FABRIC V2 的实质变化是：保留单个 motif-event identity；让 DNA/RNA events 在标量化前共同进入 gene graph；先得到完整 path vector，再由共享非线性 readout 产生 path logit；最后按 singleton event 或完整 model-injection-equivalence set 提供到 local alternative 和 full path 的 gauge-invariant counterfactual output。

FABRIC V2 的 per-epoch validation 只保留两个核心数值：用于选择的 molecule-weighted compatible-path NLL，以及只作报告的 ONT-count-weighted精确 log-space matrix KL。后者不是新模型模块，也不复刻 PRISM 的 top-1/CE/KL 指标集合；在 `OntObservationProcessAudit` 证明 matrix quantifier 与 compatible likelihood 使用同一 molecule population 前，它只表示 same-library cross-pipeline agreement。

### 19.2 Otari

[Otari](../paper/Otari.pdf) 是最接近的结构性先例，应在 related work 中明确讨论。Otari 为每条 transcript 单独构建 directed graph，以 ConvSplice、Sei 和 Seqweaver 的静态 sequence-predicted features 作为 node attributes，graph pooling 后独立回归 30 个 tissues 的 \(\log_2(TPM+0.01)\)，并使用 MSE 与 triplet objective；variant analysis 通过改变序列并重算特征完成。TPM 是 library-normalized expression，不是绝对分子丰度，也不是同一 gene 内归一化的 isoform usage probability。FABRIC V2 则在每个 gene 的共享图上竞争 ONT-matrix structural paths，使用细胞特异的 mapped ATAC 与 factor/RBP activities，采用 compatible-path likelihood，并以具名 singleton factor-motif-cell event 或完整 injection-equivalence set 的 local contrast 与 path/group redistribution 为主要解释单位。

FABRIC 不把“GNN + 三块特征 + pooling”本身写成架构创新。方法贡献是不同的 estimand、数据生成过程、合法路径竞争、compatibility-aware observed-molecule likelihood 和 event-to-choice-to-path interpretation。

## 20. Code modification boundary

实施 V2 时保留 `graph.py` 中的 per-gene graph、精确 local adjacency、ordered legal paths 与 sparse path incidence，保留 `likelihood.py` 中的 compatible-path likelihood。`motifs.py` 只保留已经审计的 motif parser/source validation、factor/group provenance、strand/coordinate projection、确定性 scan/rank/tie-break，以及 saturation/boundary 审计不变量；V1 的 choice-region builder、把 `choice_id`/alternative relation 写进 event identity、choice-based cap 和旧 event-vector assembly 必须替换。新增 `motif_equivalence_family` collapse、accessibility-only/group-activity schema、`PhysicalEventTable`、`EventRouteTable` 和本合同的 graph-anchor cap/routing；同时保留 pre-cap candidate-route identity 与 cap decision以生成 split-neutral `RouteDegreeCapAudit`，final model route table 仍只含 retained routes。`motifs.py` 只执行 biological physical collapse，并验证 `should_have_collapsed_error` 不存在；不得按 model signature 去重。missingness、cell-level ATAC `mapping_valid` 和 train-only gate baseline 的核心语义保留；ESS/evenness/coverage/distance/consistency 仅进入映射审计与分层报告，不进入模型 tensors；按本合同增加 post-raw z-score、shared gate keys、support-domain flags 和 active-event filtering。浅层 GraphGPS 复用最小实现。

`dataset.py`需要构造唯一`(gene, cell)` instances、全部active named-event tensors、shared gate-key tensors、仅含post-cap production weights的冻结routing matrix、reference-coded base features、raw-cell supported-rectangle canonical interaction basis、raw contrast/comparator table、`InteractionSupportManifest`、final encoded tensors上的`ModelInjectionEquivalenceIndex`、`GateCollinearityAudit`、`LongReadCompatibilityAudit`、`RNAWindowCoverageAudit`、RETAINED_INTRON evidence/support fields、route-degree/cap与token/path-scale audit strata及三块model input；pre-cap reference weights不进入模型。它必须验证并消费 §3.3 的外部 compatible-EC artifact/manifest，拒绝以历史 7,198-gene input 补齐 17,706-gene catalog。`model.py`维持当前 V2 event aggregation、GraphGPS 与 path readout。`evaluate.py`实现机制解释与正式评价，并为 §15.4 提供唯一的精确 log-space count-weighted ONT KL。`train.py`负责统一单任务 runtime、逐命令 hyperparameter resolution、per-gene capped sampling、gene-complete gradient clipping 与 optimizer step、validation-NLL scheduler、每 epoch 一次 validation traversal，以及两个核心 validation 数值；它不完整评估 train、不自动展开 multi-run matrix，也不在训练中计算 test predictions/metrics。

V2 实施后，失去调用者的 V1 internal classes 直接删除，不保留 `legacy_*`、`*_v2` wrapper、factory、registry、fallback chain 或旧 checkpoint migration。V1 历史由 Git、V1 文档和既有 artifacts 保存。

## 21. Explicit non-goals

V2 当前主模型明确不加入第二张 ChoiceGraph、pairwise CRF、choice-pair catalog、skip-pair discovery、三阶 factor、显式 PPI、factor/event-specific nonlinear dose-response layer、第二个 per-token event cap、event-count normalization、event attention、event dropout、GraphGPS stack、path Transformer、autoregressive transcript decoder、sequence CNN、AlphaGenome fine-tuning 或未经 §4.1 独立冻结的 AlphaGenome 输入、独立 State 解释分支、path-ID parameters、nascent-RNA latent state、IR-specific prediction head、降解生成模型、auxiliary losses、Shapley attribution、raw FASTQ re-alignment、BAM regeneration、GTF reannotation/transcript discovery、通用 artifact framework 或插件系统。§3.3 从冻结既有 alignments 与冻结 matrix structural-path catalog 提取 compatible-EC 的外部数据准备步骤是训练监督的必需前置，不属于这些 non-goals，也不成为模型 runtime。当前 routing 也不启用 route broadcast、\(1/\sqrt{degree}\) correction、learned/adaptive route weights、site-carrier token 或 runtime/per-event routing switch；`ModelInjectionEquivalenceIndex` 不启用 model-signature forward deduplication。

任何新增机制必须说明它表达了当前模型无法表达且数据能够辨识的具体生物学关系，并经过用户明确批准后修改本合同。性能不足、未来可能有用或已有论文使用，不构成自动增加复杂度的理由。

## 22. Implementation sequence and admission

- [ ] **单任务资源估算已冻结**：每个拟启动命令必须基于相同 condition、model shape、gene-cell cap、gene-shape GPU policy 与 shard-prefetch policy 的 train/validation-only full-shape pilot，记录一次运行的 GPU-hours、峰值 GPU memory、host RAM、prepared/runtime storage、完整 validation 耗时、预计 epoch 数、early-stopping patience、wall-clock 和失败重跑预留。profile 若受到其他 GPU workload 竞争则 timing 不可冻结。profiling 不执行 `optimizer.step`、gradient clipping、不保存 checkpoint、不完成真实 full epoch，也不估算或授权隐藏的 multi-seed/multi-condition campaign；其耗时必须明确标为不含 optimizer-update/clipping 的投影。condition、model shape、gene-cell cap、batch/precision/prefetch policy 等 resource-relevant identity 变化后旧 profile 失效；仅 learning rate 或 validation-driven scheduler 改变不伪造新的 batch-shape profile。

V2实施按一个最小闭环推进。第一项外部前置是从既有 ONT alignments 和 matrix structural paths 交付覆盖 17,706 structural candidates 的 compatible-EC artifact；在整数 mass conservation、split exposure 与 artifact admission 通过前不能进入真实 cohort 训练。随后冻结 graph/path/event/routing、train-only normalization/admission、interaction design、17,600-gene G_fit、validation-only ONT count target 与 `(gene, cell)` batch contract。toy/fixture 必须覆盖 forward、compatible NLL、逐 gene optimizer step、gene 内 microbatch accumulation、单次 validation traversal和精确 count-weighted ONT matrix KL；真实 fixture 继续覆盖坐标、strand、gate、route、path identity、scope/exclusion conservation及关键机制审计。最后才完成与当前 runtime 身份一致的 GPU memory/throughput profile 和资源冻结；训练与 test 仍分别受独立授权控制。

若 held-out test model inference 尚未首次运行，但某个独立 command 在 train/validation 上触发预冻结的 `PathScaleAudit` gate，只能先把该 command 标为 architecture diagnostic、修订合同并重新运行受影响的 commands；不得保留双 runtime。若 `RouteDegreeCapAudit` 的 `implementation_valid=false`，implementation admission 直接失败；若 baseline 或 catalog-applicable capability gate 失败，当前 routing 不得获得 training admission。任何替代 routing 必须先单独修订合同，并由用户用明确 seed/condition 命令重新运行；不得增加 inference-time 或 per-event routing switch。

实现测试通过、输入与 manifests 冻结、与该命令 resource-relevant identity 匹配的 profile 完成且 `training_authorized=true` 后，才允许启动所选的一次 full-cohort run。`TrainingRunManifest` 在启动时记录该命令唯一的 seed、condition 以及全部 resolved optimizer/training controls。checkpoint 冻结后，只有另行设置 `final_test_authorized=true` 才能运行依赖 held-out predictions 的 diagnostics；任何失败只撤回它所约束的 claim，不能反馈选择 seed、condition、scaling 或 routing。AlphaGenome deferred extension 不阻塞当前 V2 主模型实现，也不能在该阶段被顺手启用。

本文档本身不授权训练，也不把通过单元测试等同于科学可用。代码、配置、真实 artifacts、测试、资源 profiling、训练授权与 held-out test 状态必须分别报告。

## 23. Final definition

> **FABRIC V2 是一个 cell-conditioned、event-resolved、gene-level legal-path model：具名 factor–DNA/RNA motif events、accessibility-only events 和静态 CIS 共同在一张 gene processing graph 中上下文化，完整 matrix isoform paths 在向量汇总后竞争 matrix-catalog-compatible observed-library conditional probability，局部 alternative effects 从 path distribution 中边缘化，singleton event 或完整 model-injection-equivalence set 的影响以 gauge-invariant 模型反事实定义；AlphaGenome `embeddings_1bp` 仅是尚未启用的后续静态 CIS 扩展。**
