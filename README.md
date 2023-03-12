# On_the_Utility_of_Synthetic_Data
Code release for "A New Benchmark: On the Utility of Synthetic Data with Blender for Bare Supervised Learning and Downstream Domain Adaptation", accepted by CVPR2023.

## Requirements
- python 3.8.8
- pytorch 1.8.0
- torchvision 0.9.0

## Data preparation
1. Existing datasets. The references of the used exsiting datasets (e.g., VisDA-2017, ImageNet, and MetaShift) are included in the paper.
2. Our new datasets. SynSL is here; SynSL-120K is here; S2RDA is [here](https://pan.baidu.com/s/17C5lRDf7cpGR1kAVS2jS-Q?pwd=61tt).

## Model training
1. Install necessary python packages.
2. Replace data paths in run.sh with those in one's own system. 
3. Run command `sh run.sh`.

The results are saved in the folder `./checkpoints/`.

## Paper citation
```
@InProceedings{tang2023a,
    author    = {Tang, Hui and Jia, Kui},
    title     = {A New Benchmark: On the Utility of Synthetic Data with Blender for Bare Supervised Learning and Downstream Domain Adaptation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
}
```
