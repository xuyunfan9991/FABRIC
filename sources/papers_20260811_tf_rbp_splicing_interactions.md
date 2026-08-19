# TF-RBP protein interactions and alternative splicing: evidence and FABRIC scope

Date: 2026-08-11

## WT1(+KTS)-U2AF65/U2AF2

### Primary evidence

- Davies et al. *Genes & Development* (1998), “WT1 interacts with the splicing factor U2AF65 in an isoform-dependent manner and can be incorporated into spliceosomes.”
  - PMID: 9784496
  - DOI: 10.1101/gad.12.20.3217
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/9784496/
  - Yeast two-hybrid, GST pull-down, co-IP and localization experiments support a WT1-U2AF65 interaction. The +KTS isoform interacts more strongly and WT1 can enter spliceosomes assembled on a model pre-mRNA.
  - Important limit: this study did not identify a specific endogenous RNA target or establish an event-specific alternative-splicing effect of the WT1-U2AF2 complex.

- Larsson et al. *Cell* (1995), “Subnuclear localization of WT1 in splicing or transcription factor domains is regulated by alternative splicing.”
  - PMID: 7736591
  - DOI: 10.1016/0092-8674(95)90392-5
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/7736591/
  - Supports isoform-dependent nuclear localization and association with spliceosomal components, but not a named target event or causal direction.

## SPI1/PU.1-NONO/p54nrb

### Primary evidence

- Hallier et al. *Journal of Biological Chemistry* (1996), “The transcription factor Spi-1/PU.1 binds RNA and interferes with the RNA-binding protein p54nrb.”
  - PMID: 8626664
  - DOI: 10.1074/jbc.271.19.11177
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/8626664/
  - Biochemical assays support SPI1-NONO interaction, SPI1 RNA binding, interference with NONO RNA binding, and altered splicing in an in-vitro assay.
  - Important limit: this does not establish a locus-specific, endogenous SPI1-NONO alternative-splicing mechanism in vivo.

- Théoleyre et al. *Oncogene* (2004), “Spi-1/PU.1 but not Fli-1 inhibits erythroid-specific alternative splicing of 4.1R pre-mRNA in murine erythroleukemia cells.”
  - PMID: 14647452
  - DOI: 10.1038/sj.onc.1207206
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/14647452/
  - Supports an endogenous SPI1-dependent change in 4.1R exon 16 inclusion, but does not establish NONO as the mediator.

- Guillouf et al. *Journal of Biological Chemistry* (2006), “Spi-1/PU.1 oncoprotein affects splicing decisions in a promoter binding-dependent manner.”
  - PMID: 16698794
  - DOI: 10.1074/jbc.M512049200
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/16698794/
  - Reporter and mutant experiments show that promoter-bound/transactivating SPI1 can change alternative 5-prime splice-site choice. This is evidence for transcription-splicing coupling, not direct proof of a NONO-mediated route.

## Modeling conclusion for FABRIC V1

The physical interactions are biologically plausible, but they are not identifiable from DNA/RNA sequence, mapped ATAC accessibility and gene-level RNA activity alone. Missing observables include protein isoform abundance, protein localization, stoichiometry, post-translational state, complex formation and locus-specific complex occupancy.

FABRIC V1 should therefore model sequence-anchored evidence rather than protein-name categories:

- DNA event: local DNA motif x ATAC accessibility x factor activity/reliability.
- RNA event: local pre-mRNA motif/sequence feature x factor activity.
- TF-RBP physical interaction: outside the named mechanistic attribution scope.

A dual-role protein may have both DNA and RNA events if each has an independent local sequence anchor. A known PPI may be added as post-hoc annotation, but not as a fitted event contribution. A large foundation or world model could provide a prior that two proteins can interact; it cannot identify when, where or how strongly that complex acted in a particular cell-choice-path observation.

Future interaction modeling would require isoform-aware protein/activity measurements, locus-specific occupancy or CLIP evidence, and preferably single and double perturbations with a long-read splicing readout. Until then, a pairwise expression-product term would be confounded with cell state and should not be interpreted as physical interaction.
