#!/bin/bash - feature extraction

python main.py  --source_data_path /data/S2RDA-49/synthetic/  --target_data_path /data/S2RDA-49/real/  --num_classes 49  --source synthetic  --target real-49  --arch resnet50  --pret_stat ../no_adaptation/checkpoints/synthetic2real-49_bs64_lr0.0001_resnet50_pretrainTrue/model_best_prec1.pth.tar  --log ./checkpoints/no_adaptation/

python main.py  --source_data_path /data/S2RDA-MS-39/synthetic/  --target_data_path /data/S2RDA-MS-39/real/  --num_classes 39  --source synthetic  --target real-ms-39  --arch resnet50  --pret_stat ../no_adaptation/checkpoints/synthetic2real-ms-39_bs64_lr0.0001_resnet50_pretrainTrue/model_best_prec1.pth.tar  --log ./checkpoints/no_adaptation/


python main.py  --source_data_path /data/S2RDA-49/synthetic/  --target_data_path /data/S2RDA-49/real/  --num_classes 49  --source synthetic  --target real-49  --arch resnet50  --pret_stat ../SRDC/checkpoints/synthetic2real-49_bs64_lr0.0001_resnet50_pretrainTrue/model_best_prec1.pth.tar  --log ./checkpoints/SRDC/

python main.py  --source_data_path /data/S2RDA-MS-39/synthetic/  --target_data_path /data/S2RDA-MS-39/real/  --num_classes 39  --source synthetic  --target real-ms-39  --arch resnet50  --pret_stat ../DisClusterDA/checkpoints/synthetic2real-ms-39_bs64_lr0.0001_resnet50_pretrainTrue/model_best_prec1.pth.tar  --log ./checkpoints/DisClusterDA/
