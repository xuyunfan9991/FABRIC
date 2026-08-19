# CTCF-CD45/PTPRC exon 5: literature evidence for a FABRIC positive-control case

Date: 2026-08-11

## Primary studies

1. Shukla et al. *Nature* (2011), “CTCF-promoted RNA polymerase II pausing links DNA methylation to splicing.”
   - PMID: 21964334
   - DOI: 10.1038/nature10442
   - PubMed: https://pubmed.ncbi.nlm.nih.gov/21964334/
   - Key evidence: CTCF binds the DNA at the alternatively spliced CD45/PTPRC exon 5; CTCF depletion or mutation of the exon-5 CTCF-binding site reduces local RNA polymerase II pausing and exon 5 inclusion. DNA methylation is inversely associated with CTCF binding and exon inclusion.

2. Marina et al. *EMBO Journal* (2016), “TET-catalyzed oxidation of intragenic 5-methylcytosine regulates CTCF-dependent alternative splicing.”
   - PMID: 26711177
   - DOI: 10.15252/embj.201593235
   - PubMed: https://pubmed.ncbi.nlm.nih.gov/26711177/
   - Key evidence: TET1/TET2-dependent oxidation of intragenic 5mC facilitates CTCF association. TET depletion increases 5mC, reduces CTCF binding, and reduces CD45 exon 5 inclusion. The study also connects activation-dependent changes in CD4 T cells with exon-5-associated cytosine modification, CTCF occupancy, and splicing.

## Mechanistic interpretation

Low methylation / favorable chromatin context -> CTCF occupancy at PTPRC exon 5 -> local Pol II pausing -> more time for recognition of the weak alternative exon -> increased exon 5 inclusion -> increased abundance of exon-5-containing full transcript paths.

The reverse state favors exon 5 skipping. In manuscript language, use “exon 5 inclusion/skipping,” not “intron retention.”

## What ATAC can and cannot establish

- ATAC can support that the local DNA site is accessible and compatible with factor binding.
- Motif occurrence plus accessibility does not by itself establish CTCF occupancy.
- ATAC does not directly measure Pol II pausing.
- CTCF CUT&Tag/ChIP-seq and Pol II ChIP, PRO-seq, or NET-seq provide orthogonal evidence for occupancy and pausing.
- CTCF knockdown, targeted motif disruption, methylation editing, or TET perturbation provide causal validation.

## Proposed FABRIC positive-control evidence chain

1. PTPRC is expressed and exon 5 inclusion/skipping is identifiable with adequate long-read support.
2. The mapped ATAC context is high-reliability and shows accessibility over the exon-5 CTCF site in the relevant cell state.
3. A named CTCF DNA-motif event is routed to the exon-5 local alternative.
4. Its signed contribution increases the relative logit of exon 5 inclusion versus skipping in the expected state.
5. This local contribution propagates to named exon-5-containing legal transcript paths.
6. Masking the CTCF DNA event causes a signed decrease in predicted exon-5 inclusion and in those full-path probabilities.
7. Negative controls include nearby non-CTCF events, accessibility or factor-activity permutations, and ATAC-neighbor shuffling.

This case is a mechanism-consistency positive control. Without direct occupancy or perturbation data, it should not be described as proving that ATAC-measured CTCF binding causally changes splicing.
