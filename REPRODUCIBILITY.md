\# Reproducibility Notes



This repository provides the code, frozen result files, source CSV files, and final figures used in the experimental study of Poss-Logic-LM.



\## Scope of the Repository



The full experimentation was executed across multiple Kaggle and Colab sessions because of GPU and runtime constraints. The complete workflow involved several notebooks distributed across different execution environments.



For review and verification, this repository contains:



\- the consolidated source code used for the final pipelines,

\- the JSON result files produced by all reported variants,

\- the beam candidate databases used by Poss-Logic-LM,

\- the CSV files used to generate the reported figures,

\- the final generated figures,

\- the scripts required to recompute summary metrics from the stored JSON files.



The notebooks are not all included because they were distributed across different Kaggle sessions. However, the outputs required to verify the reported results are included.



\## Verification vs Full Re-execution



This repository supports two levels of reproducibility.



\### 1. Result verification



Reviewers can verify the reported results directly from the stored JSON and CSV files. This is the recommended path.



The main result folders are:



```text

LOGICLM-4Variants/

POSS-LOGICLM/

analysis\_outputs/final\_figures/

Figures/

