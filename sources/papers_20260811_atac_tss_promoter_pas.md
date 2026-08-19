# Local chromatin accessibility for alternative TSS/promoter and PAS usage

Date: 2026-08-11

## Summary

- Alternative TSS/promoter: local ATAC accessibility is a strong, biologically direct permissive context for promoter use. TSS usage itself still requires RNA-end evidence such as CAGE/RAMPAGE or reliable complete long-read 5-prime ends.
- TSS-to-PAS coupling: promoter choice can alter downstream splicing and 3-prime end selection, making promoter accessibility particularly useful in a full-path model.
- PAS-local ATAC: evidence is weaker and directionally complex. It is best described as chromatin/elongation context, not as direct PAS activation.

## Alternative TSS/promoter evidence

1. Buenrostro et al. *Nature Methods* (2013), “Transposition of native chromatin for fast and sensitive epigenomic profiling of open chromatin, DNA-binding proteins and nucleosome position.”
   - PMID: 24097267
   - DOI: 10.1038/nmeth.2688
   - PubMed: https://pubmed.ncbi.nlm.nih.gov/24097267/
   - Establishes ATAC-seq as an assay of open chromatin and nucleosome architecture; it is not itself a promoter-choice causal study.

2. Nepal et al. *Nature Communications* (2023), alternative promoter usage in hepatocellular carcinoma.
   - PMID: 37169774
   - DOI: 10.1038/s41467-023-38272-4
   - PubMed: https://pubmed.ncbi.nlm.nih.gov/37169774/
   - CAGE-defined alternative promoters are supported by ATAC, Pol II and methylation profiles. This is strong association and mechanistic context, but not direct accessibility manipulation at each promoter.

3. Zhang et al. *Nature Communications* (2023), IRF5 alternative-promoter regulation.
   - PMID: 36869052
   - DOI: 10.1038/s41467-023-36897-z
   - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9984425/
   - ATAC identifies accessible regulatory regions; chromatin-contact assays, CRISPR perturbation, nucleotide editing and promoter-specific RNA readouts provide a stronger causal chain.

4. Greenberg et al. *eLife* (2019), Zdbf2 promoter switching.
   - PMID: 30990414
   - DOI: 10.7554/eLife.44057
   - Article: https://elifesciences.org/articles/44057
   - State-dynamic accessible elements were followed by enhancer deletion, CTCF-site perturbation, chromatin-contact assays and promoter-specific readouts.

5. Hou et al. *Nature Communications* (2023), CamoTSS.
   - DOI: 10.1038/s41467-023-42636-1
   - Article: https://www.nature.com/articles/s41467-023-42636-1
   - Uses ATAC-supported open regions to help identify high-confidence alternative TSS clusters from 5-prime single-cell RNA-seq.

## PAS and chromatin evidence

1. Lee and Chen. *Bioinformatics* (2013), “Alternative polyadenylation sites reveal distinct chromatin accessibility and histone modification in human cell lines.”
   - PMID: 23740743
   - DOI: 10.1093/bioinformatics/btt288
   - PubMed: https://pubmed.ncbi.nlm.nih.gov/23740743/
   - PAS regions were generally less DNase-sensitive; distal PAS could be even less accessible. Associations were not a simple positive accessibility-usage relationship.

2. Spies et al. *Molecular Cell* (2009), nucleosome positioning around polyadenylation sites.
   - PMID: 19854133
   - DOI: 10.1016/j.molcel.2009.10.008
   - PubMed: https://pubmed.ncbi.nlm.nih.gov/19854133/

3. Geisberg et al. *PNAS* (2024), “Chromatin regulates alternative polyadenylation via the RNA polymerase II elongation rate.”
   - PMID: 38748572
   - DOI: 10.1073/pnas.2405827121
   - PubMed: https://pubmed.ncbi.nlm.nih.gov/38748572/
   - Perturbation of chromatin factors, histones and Pol II speed in yeast supports a chromatin-to-elongation-to-APA mechanism, rather than a direct ATAC-peak-to-PAS mechanism.

4. Nanavaty et al. *Molecular Cell* (2020), methylation/CTCF/cohesin control of alternative polyadenylation.
   - PMID: 32333838
   - DOI: 10.1016/j.molcel.2020.03.024
   - PubMed: https://pubmed.ncbi.nlm.nih.gov/32333838/
   - Provides locus-level causal evidence involving methylation, CTCF and cohesin looping; ATAC alone does not observe those mediators.

5. Alfonso-Gonzalez et al. *Cell* (2023), “Sites of transcription initiation drive mRNA isoform selection.”
   - PMID: 37178687
   - DOI: 10.1016/j.cell.2023.04.012
   - PubMed: https://pubmed.ncbi.nlm.nih.gov/37178687/
   - Long reads and promoter perturbation demonstrate widespread coupling of TSS choice to downstream splicing and 3-prime end choice.

## FABRIC modeling implication

The clean factor-independent ATAC term is a promoter-accessibility event:

- define alternative promoter clusters, not individual nearby TSS bases;
- assign a fixed local promoter window and cell/state-specific accessibility;
- center its contribution within the TSS choice;
- propagate the TSS contribution to every legal full transcript path beginning at that promoter;
- use accessibility masking and matched-state ATAC shuffling as counterfactual controls.

This should be a separate accessibility main effect. TF-motif x ATAC terms should quantify factor-specific evidence beyond general promoter openness, avoiding forced attribution of generic accessibility to whichever motif happens to be present.

For PAS, use RNA polyadenylation-signal sequence and cleavage/polyadenylation-factor evidence as primary local features. PAS-local ATAC, if used, should be an anonymous strand-aware upstream/site/downstream chromatin-context feature with no fixed positive direction. The strongest main-text route is promoter accessibility -> TSS choice -> downstream full-path/PAS distribution.

ATAC does not establish transcription initiation, cleavage, protein occupancy or causality on its own. Standard long-read RNA can also have 5-prime truncation and internal-priming artifacts, so terminal choices require independent endpoint support and identifiability checks.
