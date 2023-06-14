# Blazed oblique plane microscopy reveals scale-invariant inference of brain-wide population activity


The `notebooks` folder contains a jupyter notebook that downloads the data of all animals used in the paper and   generates the main figure panels in the paper from the cell coordinates and delta F/F traces for one of them. 

The notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danionella/hoffmann_et_al_2023/)] also contains code that sets up a suitable environment in google collab and were tested with a google collab pro high memory instance. In case there is a warning about multiple installed and conflicting cupy version this can be ignored. 

The high memory instance is only necessary to compute the pairwise correlations in `Figure_3`, otherwise a regular collab instance is sufficient to perform the analysis. 