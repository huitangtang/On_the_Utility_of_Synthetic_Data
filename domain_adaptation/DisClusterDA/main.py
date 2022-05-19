######################################################################################
#                                                                                    #
# All the codes about the model construction should be kept in the folder ./models/. #
# All the codes about the data processing should be kept in the folder ./data/.      #
# The file ./opts.py stores the options.                                             #
# The file ./trainer.py stores the training and test strategies.                     #
# The ./main.py should be simple.                                                    #
#                                                                                    #
######################################################################################
import os
import json
import shutil
import torch
import torch.optim
import torch.nn as nn
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time
import ipdb
import copy

from models.resnet import ResBase # for the model construction
from trainer import train_compute_class_mean # for the training process
from trainer import evaluate # for the evaluation process
from opts import opts # options for the project
from data.prepare_data import generate_dataloader # prepare data and dataloader
from utils.EntropyMinimizationLoss import AdaptiveFilteringEMLossForTarget # adaptive filtering entropy minimization loss (target)
from utils.consistency_loss import ConsistencyLoss # consistency loss (target)


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
    
    current_epoch = 0
    
    # define base model
    model = ResBase(option=args.arch, pret=args.pretrained, num_classes=args.num_classes, num_neurons=args.num_neurons)
    # define multi-GPU
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion_set = {'ce': nn.CrossEntropyLoss().cuda(), 'afem': AdaptiveFilteringEMLossForTarget(eps=args.eps).cuda(), 'con': ConsistencyLoss(div=args.div).cuda()}
        
    # apply different learning rates to different layers
    lr_fe = args.lr * 0.1 if args.pretrained else args.lr
    params_list = [
            {'params': model.module.features.parameters(), 'lr': lr_fe},
            {'params': model.module.fc1.parameters()},
    ]
    cen_set = {'cs_1': Variable(torch.cuda.FloatTensor(args.num_classes, model.module.feat1_dim).fill_(0)), 
               'ct_1': Variable(torch.cuda.FloatTensor(args.num_classes, model.module.feat1_dim).fill_(0))}
    if args.num_neurons is not None:
        params_list.append({'params': model.module.fc2.parameters()})
        cen_set['cs_2'] = Variable(torch.cuda.FloatTensor(args.num_classes, model.module.feat2_dim).fill_(0))
        cen_set['ct_2'] = Variable(torch.cuda.FloatTensor(args.num_classes, model.module.feat2_dim).fill_(0))
        
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params_list,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov)
    
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
        print("==> loading checkpoints '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        current_epoch = checkpoint['epoch']
        best_accs = checkpoint['best_accs']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("==> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()

    # start time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------')
    log.close()

    cudnn.benchmark = True
    # process data and prepare dataloaders
    train_loader_source, train_loader_target, val_loader_target, val_loader_source = generate_dataloader(args)
    
    if args.eval_only:
        prec1 = evaluate(val_loader_target, model, criterion_set['ce'], -1, args)
        print(' * Eval acc@1: {:.3f}'.format(prec1))
        return

    print('begin training')
    train_loader_source_batch = enumerate(train_loader_source)
    train_loader_target_batch = enumerate(train_loader_target)
    batch_number = count_epoch_on_large_dataset(train_loader_target, train_loader_source)
    start_itern = current_epoch * batch_number
    num_itern_total = args.epochs * batch_number
    test_freq = int(num_itern_total / 200)
    print('test_freq: ', test_freq)
    
    train_records = {'batch_time': AverageMeter(),
                    'data_time': AverageMeter(),
                    'losses': AverageMeter(),
                    'top1_s': AverageMeter(),}
    for itern in range(start_itern, num_itern_total):
        # train for one iteration
        train_loader_source_batch, train_loader_target_batch, cen_set = train_compute_class_mean(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model, criterion_set, optimizer, itern, current_epoch, cen_set, args, train_records)
        # evaluate on target
        if (itern + 1) % batch_number == 0 or (itern + 1) % test_freq == 0:
            accs = evaluate(val_loader_target, model, criterion_set['ce'], current_epoch, args)
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
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_accs': best_accs,
                }, is_best, args, suffix='_' + k)
            
            # update learning rate
            if (itern + 1) % batch_number == 0:
                scheduler.step()
                current_epoch += 1
            
            # reset records
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
    

def count_epoch_on_large_dataset(train_loader_target, train_loader_source):
    batch_number_t = len(train_loader_target)
    batch_number = batch_number_t
    batch_number_s = len(train_loader_source)
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





