# Blazed oblique plane microscopy reveals scale-invariant inference of brain-wide population activity


The `notebooks` folder contains a jupyter notebook that downloads all calcium-dependent fluorescence traces used in the paper and generates the data panels of figures 3 to 5. 


### Colab
The notebook contains code that sets up a suitable environment in Google Colab (click here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danionella/hoffmann_et_al_2023/blob/main/notebooks/generate_figures.ipynb)). In case there is a warning about conflicting cupy versions, it can be ignored. 

A Colab Pro pro instance with a A100 GPU is needed to compute the pairwise correlations in Figure 3. A free Colab GPU instance is sufficient to perform the analysis of the remaining figures 4 and 5.


### To install software and download the data locally

Using [mamba](https://github.com/conda-forge/miniforge#mambaforge) (or conda) as package manager:
```
 mamba create -n hoffmannetal -c rapidsai -c conda-forge -c nvidia rapids=23.10 python=3.10 cudatoolkit=11.8
 mamba activate hoffmannetal 
 git clone https://github.com/danionella/hoffmann_et_al_2023.git
 pip install --ignore-installed  --quiet ./hoffmann_et_al_2023
 wget "https://gin.g-node.org/danionella/Hoffmann_et_al_2023/raw/5a3146dc108208415f87bf17ebce37d566b28208/20230611_export_3.h5" -O data.h5
```

A faster but non permanent data repository:
```
wget "https://owncloud-ext.charite.de/owncloud/index.php/s/H97Qi8haRYLZu4e/download" -O data.h5
```

NVIDIA GeForce RTX 3090


Hardware requirements: 
  For Figure 3: GPU with memory >= 32 GB
  For Figure 4 and 5: GPU with memory >= 16 GB

