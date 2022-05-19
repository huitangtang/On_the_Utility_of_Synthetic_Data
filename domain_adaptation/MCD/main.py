##############################################################################
#
# All the codes about the model constructing should be kept in the folder ./models/
# All the codes about the data process should be kept in the folder ./data/
# The file ./opts.py stores the options.
# The file ./trainer.py stores the training and test strategy
# The ./main.py should be simple
#
##############################################################################
import json
import os
import shutil
import time
import math
import random
import numpy as np
import copy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from data.prepare_data import generate_dataloader  # prepare the data and dataloader
from models.resnet import ResBase, ResClassifier  # model construction
from opts import opts  # options for the project
from trainer import train  # for the training process
from trainer import validate  # for the validation (test) process
from models.EntropyMinimizationLoss import EMLossForTarget
import ipdb

best_accs = {'prec1': 0, 'mcp': 0}

def main():
    global args, best_accs
    current_epoch = 0
    args = opts()
    
    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    
    if args.arch.find('resnet') != -1:
        G = ResBase(option=args.arch, feat_proj=args.feat_proj, pret=args.pretrained, num_classes=args.num_classes)
        ############  !!!! the middle = 1000 follows the MCD setting. !!!!!!!###############
        F1 = ResClassifier(num_classes=args.num_classes, num_layer=args.num_fc, num_unit=G.dim, prob=0.5, middle=1000)
        F2 = ResClassifier(num_classes=args.num_classes, num_layer=args.num_fc, num_unit=G.dim, prob=0.5, middle=1000)
    else:
        raise ValueError('Unavailable model architecture!!!')
        
    # define multi-GPU
    G = torch.nn.DataParallel(G).cuda()
    F1 = torch.nn.DataParallel(F1).cuda()
    F2 = torch.nn.DataParallel(F2).cuda()
    model_set = {'G': G, 'F1': F1, 'F2': F2}
    
    # define loss function (criterion) and optimizer
    criterion_ce = nn.CrossEntropyLoss().cuda()
    criterion_em_target = EMLossForTarget().cuda()
    criterion_set = {'ce': criterion_ce, 'em': criterion_em_target}
    
    # apply different learning rates to different layers
    lr_fe = args.lr * 0.1 if args.pretrained else args.lr
    if args.arch.find('resnet') != -1:
        params_list = [
                {'params': G.module.features.parameters(), 'lr': lr_fe},
        ]
        if args.feat_proj:
            params_list.append({'params': G.module.proj_fc.parameters()})
        optimizer_G = torch.optim.SGD(params_list,
            lr=args.lr,
            # momentum=args.momentum,  ### it is commented in MCD.
            weight_decay=args.weight_decay,
            nesterov=False)
        optimizer_F = torch.optim.SGD([
            {'params': F1.parameters()},
            {'params': F2.parameters()}
        ],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=False)
        optimizer_set = {'G': optimizer_G, 'F': optimizer_F}
    else:
        raise ValueError('Unavailable model architecture!!!')
    
    # learning rate decay scheduler
    if args.lr_scheduler == 'cosine':
        scheduler_set = {k: torch.optim.lr_scheduler.CosineAnnealingLR(v, T_max=args.epochs, eta_min=0, last_epoch=-1) for k, v in optimizer_set.items()}
    else:
        if args.lr_scheduler == 'dann':
            lr_lambda = lambda epoch: 1 / pow((1 + 10 * epoch / args.epochs), 0.75)
        elif args.lr_scheduler == 'step':
            lr_lambda = lambda epoch: args.gamma ** (epoch + 1 > args.decay_epoch[1] and 2 or epoch + 1 > args.decay_epoch[0] and 1 or 0)
        elif args.lr_scheduler == 'const':
            lr_lambda = lambda epoch: 1
        scheduler_set = {k: torch.optim.lr_scheduler.LambdaLR(v, lr_lambda, last_epoch=-1) for k, v in optimizer_set.items()}

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
        now_state_dict = model_set['G'].state_dict()
        now_state_dict.update(pret_stat_)
        model_set['G'].load_state_dict(now_state_dict)
        print("==> loaded pretrained state '{}' (iter {})"
              .format(args.pret_stat, checkpoint['iter']))
        
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            current_epoch = checkpoint['epoch']
            best_accs = checkpoint['best_accs']
            for k in model_set.keys():
                model_set[k].load_state_dict(checkpoint['model_state_dict'][k])
            for k in optimizer_set.keys():
                optimizer_set[k].load_state_dict(checkpoint['optimizer_state_dict'][k])
                scheduler_set[k].load_state_dict(checkpoint['scheduler_state_dict'][k])
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
    # process the data and prepare the dataloaders
    source_train_loader, source_val_loader, target_train_loader, val_loader = generate_dataloader(args)
    # test only
    if args.test_only:
        validate(val_loader, model_set, -1, args)
        return
    # start time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------')
    log.close()
    
    print('begin training')
    source_train_loader_batch = enumerate(source_train_loader)
    target_train_loader_batch = enumerate(target_train_loader)
    batch_number = count_epoch_on_large_dataset(target_train_loader, source_train_loader)
    start_itern = current_epoch * batch_number
    num_itern_total = args.epochs * batch_number
    args.test_freq = int(num_itern_total / 200)
    print('test_freq: ', args.test_freq)
    
    train_records = {'batch_time': AverageMeter(),
                    'data_time': AverageMeter(),
                    'losses_s1': AverageMeter(),
                    'losses_s2': AverageMeter(),
                    'losses_s3': AverageMeter(),
                    'top1_c1': AverageMeter(),
                    'top1_c2': AverageMeter(),}
    for itern in range(start_itern, num_itern_total):
        # train for one iteration
        source_train_loader_batch, target_train_loader_batch = train(source_train_loader, source_train_loader_batch, target_train_loader, target_train_loader_batch, model_set, criterion_set, optimizer_set, itern, current_epoch, train_records, args)
        # evaluate on the val data
        if (itern + 1) % batch_number == 0 or (itern + 1) % args.test_freq == 0:
            accs = validate(val_loader, model_set, current_epoch, args)
            # record the best prec1
            for k in best_accs.keys():
                is_best = accs[k] > best_accs[k]
                if is_best:
                    best_accs[k] = accs[k]
                    log = open(os.path.join(args.log, 'log.txt'), 'a')
                    log.write('\n                                                                         best ' + k + ': %3f' % (best_accs[k]))
                    log.close()
                
                # save checkpoint
                save_checkpoint({
                    'epoch': current_epoch,
                    'arch': args.arch,
                    'best_accs': best_accs,
                    'model_state_dict': {k: v.state_dict() for k, v in model_set.items()},
                    'optimizer_state_dict': {k: v.state_dict() for k, v in optimizer_set.items()},
                    'scheduler_state_dict': {k: v.state_dict() for k, v in scheduler_set.items()},
                }, is_best, args, suffix='_' + k)
            
            # update learning rate
            if (itern + 1) % batch_number == 0:
                scheduler_set['G'].step()
                scheduler_set['F'].step()
                current_epoch += 1
            
            for k in train_records.keys():
                train_records[k].reset()
            
        if current_epoch > args.stop_epoch:
            break

    # end time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    for k in best_accs.keys():
        log.write('\n * best ' + k + ': %3f' % best_accs[k])
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------\n')
    log.close()


def count_epoch_on_large_dataset(target_train_loader, source_train_loader):
    batch_number_t = len(target_train_loader)
    batch_number = batch_number_t
    batch_number_s = len(source_train_loader)
    if batch_number_s > batch_number_t:
        batch_number = batch_number_s
    
    return batch_number


def save_checkpoint(state, is_best, args, filename='checkpoint', prefix='', suffix=''):
    save_file = os.path.join(args.log, '{}.pth.tar'.format(prefix + filename))
    torch.save(state, save_file)
    if is_best:
        shutil.copyfile(save_file, os.path.join(args.log, 'model_best{}.pth.tar'.format(suffix)))
        
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()





