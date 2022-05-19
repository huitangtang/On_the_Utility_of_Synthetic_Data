# DisClusterDA
Code release for Unsupervised Domain Adaptation via Distilled Discriminative Clustering published in Pattern Recognition.

The paper is available [here](https://authors.elsevier.com/a/1elDO77nKgiag).

## Requirements
- python 3.6.4
- pytorch 1.4.0
- torchvision 0.5.0

## Data preparation
The structure of the used datasets is shown in the folder `./data/datasets/` [here](https://github.com/huitangtang/ViCatDA). 

The original datasets can be downloaded [here](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md).

## Model training
1. Replace paths and domains in run.sh with those in one's own system. 
2. Install necessary python packages.
3. Run command `sh run.sh`.

The results are saved in the folder `./checkpoints/`.

## Article citation
```
@article{DisClusterDA,
author = {Hui Tang and Yaowei Wang and Kui Jia},
title = {Unsupervised domain adaptation via distilled discriminative clustering},
journal = {Pattern Recognition},
volume = {127},
pages = {108638},
year = {2022},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2022.108638},
url = {https://www.sciencedirect.com/science/article/pii/S0031320322001194},
}
```
