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
from trainer import train  # for the training process
from trainer import validate  # for the validation (test) process
from trainer import adjust_learning_rate
import ipdb

best_accs = {'prec1': 0, 'mcp': 0}

def main():
    global args, best_accs
    args = opts()
    
    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    
    if args.arch.find('resnet') != -1:
        model = ResBase(option=args.arch, pret=args.pretrained, num_classes=args.num_classes)
    else:
        raise ValueError('Unavailable model architecture!!!')
    
    # define-multi GPU
    model = torch.nn.DataParallel(model).cuda()
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    
    # apply different learning rates to different layers
    lr_fe = args.lr * 0.1 if args.pretrained else args.lr
    if args.arch.find('resnet') != -1:
        optimizer = torch.optim.SGD([
            {'params': model.module.features.parameters(), 'lr': lr_fe},
            {'params': model.module.fc.parameters()}
        ],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise ValueError('Unavailable model architecture!!!')
    
    # learning rate decay scheduler
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)
    else:
        if args.lr_scheduler == 'dann':
            lr_lambda = lambda epoch: 1 / pow((1 + 10 * epoch / args.epochs), 0.75)
        elif args.lr_scheduler == 'step':
            lr_lambda = lambda epoch: args.gamma ** (epoch + 1 > args.decay_epoch[1] and 2 or epoch + 1 > args.decay_epoch[0] and 1 or 0)
        elif args.lr_scheduler == 'const':
            lr_lambda = lambda epoch: 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    # optionally resume from a checkpoint
    if args.pret_stat is not None:
        print("==> loading pretrained state '{}'".format(args.pret_stat))
        checkpoint = torch.load(args.pret_stat)
        pret_stat = checkpoint['state_dict']
        pret_stat_ = copy.deepcopy(pret_stat)
        for k in pret_stat.keys():
            if k.find('fc') != -1:
                pret_stat_.pop(k)
            elif not k.startswith('module.'):
                pret_stat_['module.' + k] = pret_stat[k]
                pret_stat_.pop(k)            
        now_state_dict = model.state_dict()
        now_state_dict.update(pret_stat_)
        model.load_state_dict(now_state_dict)
        print("==> loaded pretrained state '{}' (iter {})"
              .format(args.pret_stat, checkpoint['iter']))
        
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_accs = checkpoint['best_accs']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("==> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('The file to be resumed from does not exist', args.resume)
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
    # start time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------')
    log.close()
    args.test_freq = len(train_loader) // (200 // args.epochs)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        #adjust_learning_rate(optimizer, epoch, args)
        best_accs = train(train_loader, val_loader, model, criterion, optimizer, epoch, best_accs, args)
        scheduler.step()
        
    # end time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    for k in best_accs.keys():
        log.write('\n * best ' + k + ': %3f' % best_accs[k])
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------\n')
    log.close()


if __name__ == '__main__':
    main()





