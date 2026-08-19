# Literature note: coupled RNA-processing choices and path models

Date: 2026-08-11

## Question

What model most directly represents dependencies among TSS, alternative exon/splice,
and PAS choices while preserving exact event-to-local-logit-to-full-path attribution?

## Primary biological evidence

1. Anvar et al. **Full-length mRNA sequencing uncovers a widespread coupling between
   transcription initiation and mRNA processing.** Genome Biology (2018).
   PMID: 29598823. DOI: 10.1186/s13059-018-1418-0.
   https://pubmed.ncbi.nlm.nih.gov/29598823/
   - Full-length molecules reveal non-independent TSS, splicing, and PAS usage,
     including both proximal and distant couplings.

2. Alfonso-Gonzalez et al. **Sites of transcription initiation drive mRNA isoform
   selection.** Cell (2023). PMID: 37178687.
   DOI: 10.1016/j.cell.2023.04.012.
   https://pubmed.ncbi.nlm.nih.gov/37178687/
   - Supports TSS-dependent downstream isoform selection and includes perturbational
     evidence; motivates TSS-to-exon and TSS-to-PAS conditional factors.

3. Tilgner et al. **Comprehensive transcriptome analysis using synthetic long-read
   sequencing reveals molecular co-association of distant splicing events.**
   Nature Biotechnology (2015). PMID: 25985263.
   DOI: 10.1038/nbt.3242.
   https://pubmed.ncbi.nlm.nih.gov/25985263/
   - Full-length molecules distinguish co-associated, mutually exclusive, and
     approximately independent pairs of distant alternative exons.

4. Herzel et al. **Long-read sequencing of nascent RNA reveals coupling among RNA
   processing events.** Genome Research (2018). PMID: 29903723.
   DOI: 10.1101/gr.232025.117.
   https://pubmed.ncbi.nlm.nih.gov/29903723/
   - Neighboring intron splicing states are non-independent and are linked to
     3-prime-end formation, supporting joint rather than isolated choice modeling.

5. Hardwick et al. **Single-nuclei isoform RNA sequencing unlocks barcoded
   exon connectivity in frozen brain tissue.** Nature Biotechnology (2022).
   PMID: 35256815. DOI: 10.1038/s41587-022-01231-3.
   https://pubmed.ncbi.nlm.nih.gov/35256815/
   - TSS-exon, exon-PAS, and non-adjacent exon associations can be strongly
     cell-type-specific. Pseudo-bulk coupling may therefore be induced by mixing
     cell types and must be conditioned on cell state.

6. Tilgner et al. **Microfluidic isoform sequencing shows widespread splicing
   coordination in the human transcriptome.** Genome Research (2018).
   PMID: 29196558. DOI: 10.1101/gr.230516.117.
   https://pubmed.ncbi.nlm.nih.gov/29196558/
   - Deep single-molecule reconstruction provides direct evidence for coordinated
     distant alternative-exon usage in a subset, rather than all, eligible genes.

7. Calvo-Roitberg et al. **mRNA initiation and termination are spatially
   coordinated.** Science (2025). PMID: 41066574.
   DOI: 10.1126/science.ado8279.
   https://pubmed.ncbi.nlm.nih.gov/41066574/
   - Full-length molecules support preferential upstream-TSS/upstream-PAS and
     downstream-TSS/downstream-PAS pairing, linked to chromatin organization and
     Pol II kinetics.

8. Treutlein et al. **Cartography of neurexin alternative splicing mapped by
   single-molecule long-read mRNA sequencing.** PNAS (2014). PMID: 24639501.
   DOI: 10.1073/pnas.1403244111.
   https://pubmed.ncbi.nlm.nih.gov/24639501/
   - A useful negative control: six canonical neurexin splice sites were largely
     independent across more than 25,000 full-length molecules. Coupling should be
     sparse and evidence-admitted, not imposed between all choices.

9. Fededa et al. **A polar mechanism coordinates different regions of alternative
   splicing within a single gene.** Molecular Cell (2005). PMID: 16061185.
   DOI: 10.1016/j.molcel.2005.06.035.
   https://pubmed.ncbi.nlm.nih.gov/16061185/
   - Minigene and in-vivo experiments support an upstream-to-downstream exon-choice
     mechanism modulated by promoter context and Pol II elongation.

## Relevant model precedent

10. LeGault and Dewey. **Inference of alternative splicing from RNA-Seq data with
   probabilistic splice graphs.** Bioinformatics (2013). PMID: 23846746.
   DOI: 10.1093/bioinformatics/btt396.
   https://pubmed.ncbi.nlm.nih.gov/23846746/
   - Establishes probabilistic splice-graph and identifiability ideas. A product of
     local edge probabilities is useful precedent but does not by itself represent
     arbitrary long-range choice-choice interactions.

## Modeling conclusion for FABRIC

Use a cell-conditioned factor graph / conditional random field over a gene's legal
full-length paths. Let `a_c(p)` be the alternative selected at choice `c` by legal
path `p`:

```
L_i(p) = B_cis(p)
       + sum_c phi_i,c[a_c(p)]
       + sum_(c,d in E_pair) Psi_i,cd[a_c(p), a_d(p)]

P_i(p) = exp(L_i(p)) / sum_(q in legal paths) exp(L_i(q))
```

- `phi` contains identifiable local relative-logit effects from state, promoter
  accessibility, DNA motif-factor events, and RNA motif-RBP events.
- `Psi` contains sparse choice-choice coupling: TSS-exon, TSS-PAS, exon-exon,
  or exon-PAS.
- Legal paths are hard support; invalid combinations receive probability zero.
- Exact compatible-path NLL can be retained for long-read supervision.
- Event-conditioned `Psi` terms preserve exact event-to-choice-pair-to-path
  attribution, unlike opaque propagation through a cell-conditioned GNN.
- Start with adjacent choice pairs plus a small number of data-supported long-range
  pairs. Admit a pair only when observed legal combinations and long-read support
  identify interaction beyond the two unary effects.
- Use zero-sum contrast coding for both unary and pair potentials so that a pair
  factor is a pure interaction rather than a hidden duplicate of main effects.
- Retain an explicit independent (unary-only) null model. Literature supports
  heterogeneous and sparse coupling, not universal all-to-all dependence.

## Interpretation boundary

Without perturbation, a learned pair factor is evidence for conditional coupling or
coordination, not proof that one processing decision causally changes the other.
Perturbing a promoter/TF/RBP/processing site can upgrade selected cases to a causal
claim.
