##############################################################################
#
# All the codes about the model constructing should be kept in the folder ./models/
# All the codes about the data process should be kept in the folder ./data/
# The file ./opts.py stores the options
# The file ./trainer.py stores the training and test strategy
# The ./main.py should be simple
#
##############################################################################
import json
import os
import time
import copy
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from data.prepare_data import generate_dataloader  # prepare the data and dataloader
from models.resnet import ResBase  # model construction
from opts import opts  # options for the project
from trainer import extract_feature_with_label  # feature extraction
from trainer import validate  # for the validation (test) process
import ipdb


def main():
    global args
    args = opts()
    
    if args.arch.find('resnet') != -1:
        model = ResBase(option=args.arch, pret=args.pretrained, num_classes=args.num_classes)
    else:
        raise ValueError('Unavailable model architecture!!!')
    
    # define-multi GPU
    model = torch.nn.DataParallel(model).cuda()
    
    # fix test data
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # optionally resume from a checkpoint
    if args.pret_stat is not None:
        print("==> loading pretrained state '{}'".format(args.pret_stat))
        checkpoint = torch.load(args.pret_stat)
        pret_stat = checkpoint['model_state_dict']
        pret_stat_ = copy.deepcopy(pret_stat)
        for k in pret_stat.keys():
            if k.find('fc') != -1 or k.find('classifier') != -1:
                pret_stat_.pop(k)
            elif not k.startswith('module.'):
                pret_stat_['module.' + k] = pret_stat[k]
                pret_stat_.pop(k)            
        now_state_dict = model.state_dict()
        now_state_dict.update(pret_stat_)
        model.load_state_dict(now_state_dict)
        print("==> loaded pretrained state '{}' (epoch {})"
              .format(args.pret_stat, checkpoint['epoch']))
        
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()

    cudnn.benchmark = True
    # process the data and prepare the dataloaders.
    train_loader, val_loader = generate_dataloader(args)
    #test only
    if args.test_only:
        validate(val_loader, model, criterion, -1, args)
        return
    
    extract_feature_with_label(train_loader, val_loader, model, args)


if __name__ == '__main__':
    main()





