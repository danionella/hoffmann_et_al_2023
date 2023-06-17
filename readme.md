# Blazed oblique plane microscopy reveals scale-invariant inference of brain-wide population activity


The `notebooks` folder contains a jupyter notebook that downloads the data of all recordings used in the paper and generates the main figure panels in the paper from the cell coordinates and delta F/F traces for one of them. 



### Colab
The notebook contains code that sets up a suitable environment in Google Colab (click here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danionella/hoffmann_et_al_2023/)). In case there is a warning about multiple installed and conflicting cupy version this can be ignored. 

To compute the pairwise correlations in `Figure_3` a pro-instance with an A100 GPU is needed - otherwise a free GPU-powered colab instance is sufficient to perform the analysis. 


### To install software and download the data locally

Using [mamba](https://github.com/conda-forge/miniforge#mambaforge) (or conda) as package manager:
```
 mamba create -n hoffmann_et_al
 mamba activate hoffmann_et_al
 mamba install  -q -y -c rapidsai -c conda-forge -c nvidia cucim cuml cupy
 git clone https://github.com/danionella/hoffmann_et_al_2023.git
 pip install --ignore-installed  --quiet ./hoffmann_et_al_2023
 wget "https://owncloud.charite.de/owncloud/index.php/s/zc9NTVJMw8AiuQn/download?path=%2F&files=20230611_export_3.h5" -O data.h5
```
Hardware requirements: 
  For Figure 3: GPU with memory >= 32 GB
  For Figure 4 and 5: GPU with memory >= 16 GB

