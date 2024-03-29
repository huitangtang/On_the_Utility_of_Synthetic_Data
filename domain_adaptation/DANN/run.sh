#!/bin/bash

python main.py  --epochs 20  --source_data_path /data/subvisda/train/ --target_data_path /data/subvisda/validation/  --test_data_path /data/subvisda/validation/  --num_classes 10  --source train  --target validation  --print_freq 100  --lr 0.0001  --arch resnet50  --pret_stat ../supervised_learning/checkpoints/SynSL-PreTr-fix_dataset_True-12800000imgs-resnet50-2400000iters-lr0.01-Noneiters_lr-batchsize128-aug_strong-s1/SynSL-PreTr-fix_dataset_True-12800000imgs-resnet50-2400000iters-lr0.01-Noneiters_lr-batchsize128-aug_strong-s1.pth.tar  --extra_mark SynSL
