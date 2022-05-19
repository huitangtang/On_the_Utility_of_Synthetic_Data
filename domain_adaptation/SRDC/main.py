#####################################################################################
#                                                                                   #
# All the codes about the model constructing should be kept in the folder ./models/ #
# All the codes about the data process should be kept in the folder ./data/         #
# The file ./opts.py stores the options                                             #
# The file ./trainer.py stores the training and test strategy                       #
# The ./main.py should be simple                                                    #
#                                                                                   #
#####################################################################################
import os
import json
import shutil
import torch
import random
import numpy as np
import time
import ipdb
import gc
import copy

import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from models.resnet import ResBase # for the model construction
from trainer import train, AverageMeter # for the training process
from trainer import validate, validate_compute_cen # for the validation/test process
from trainer import CLUSTER_GETTERS # for k-means clustering and its variants
from trainer import source_select # for source sample selection
from opts import opts # options for the project
from data.prepare_data import generate_dataloader # prepare the data and dataloader
from utils.consistency_loss import ConsistencyLoss # consistency loss (target)

args = opts()

best_accs = {'prec1': 0, 'mcp': 0, 'prec1_te': 0, 'mcp_te': 0, 'clu_acc': 0, 'clu_acc_2': 0}

def main():
    global args, best_accs
    
    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    
    # define model
    model = ResBase(option=args.arch, pret=args.pretrained, num_classes=args.num_classes, num_neurons=args.num_neurons)
    model = torch.nn.DataParallel(model).cuda() # define multiple GPUs
    
    # define learnable cluster centers
    cen_set = {'1': torch.cuda.FloatTensor(args.num_classes, model.module.feat1_dim).fill_(0)}
    cen_set['1'].requires_grad_(True)

    # define loss function/criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion_cons = ConsistencyLoss(div=args.div).cuda()
    
    # apply different learning rates to different layers
    lr_fe = args.lr * 0.1 if args.pretrained else args.lr
    params_list = [
            {'params': model.module.features.parameters(), 'lr': lr_fe},
            {'params': model.module.fc1.parameters()},
            {'params': cen_set['1'], 'lr': lr_fe},
    ]
    if args.num_neurons:
        cen_set['2'] = torch.cuda.FloatTensor(args.num_classes, model.module.feat2_dim).fill_(0)
        cen_set['2'].requires_grad_(True)
        params_list.extend([{'params': model.module.fc2.parameters()}, {'params': cen_set['2'], 'lr': lr_fe}])
        
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
            lr_lambda = lambda epoch: args.mu ** (epoch + 1 > args.decay_epoch[1] and 2 or epoch + 1 > args.decay_epoch[0] and 1 or 0)
        elif args.lr_scheduler == 'const':
            lr_lambda = lambda epoch: 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    
    # optionally resume from a checkpoint as a start point
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
    
    # resume from intermediate checkpoint
    epoch = 0                                
    init_state_dict = model.state_dict()
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            epoch = checkpoint['epoch']
            best_accs = checkpoint['best_accs']
            model.load_state_dict(checkpoint['model_state_dict'])
            cen_set = checkpoint['cen_set']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("==> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('The file to be resumed from does not exist!', args.resume)
    
    # make log directory
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
    train_loader_source, train_loader_target, val_loader_target, val_loader_target_t, val_loader_source = generate_dataloader(args)
    train_loader_target.dataset.tgts = list(np.array(torch.LongTensor(train_loader_target.dataset.tgts).fill_(-1))) # avoid using ground truth labels of target

    print('begin training')
    train_loader_source_batch = enumerate(train_loader_source)
    train_loader_target_batch = enumerate(train_loader_target)
    batch_number = count_epoch_on_large_dataset(train_loader_target, train_loader_source, args)
    start_itern = epoch * batch_number
    num_itern_total = args.epochs * batch_number
    test_freq = int(num_itern_total / 200)

    test_flag = False # if test, test_flag=True
    
    src_cs = torch.cuda.FloatTensor(len(train_loader_source.dataset.tgts)).fill_(1) # initialize source weights
    train_records = {'batch_time': AverageMeter(),
                    'data_time': AverageMeter(),
                    'losses': AverageMeter(),
                    'top1_s': AverageMeter(),}
    for itern in range(start_itern, num_itern_total):
        # evaluate on the target training and test data
        if itern % batch_number == 0:
            accs, c_set, f_set, l_set = validate_compute_cen(val_loader_target, val_loader_source, model, criterion, epoch, args)
            if args.tar != args.tar_t:
                accs.update(validate(val_loader_target_t, model, criterion, epoch, args))
            test_flag = True
            
            # k-means clustering or its variants
            if not itern or args.initial_cluster:
                cs_flag = (not itern and args.src_cen_first) or args.initial_cluster == 2
                cen = c_set['s'] if cs_flag else c_set['t']
                best_accs['clu_acc'], c_set['t'] = CLUSTER_GETTERS[args.cluster_method](f_set['t'], l_set['t'], l_set['t_p'] if args.cluster_method == 'kernel_kmeans' else cen, train_loader_target, epoch, model, args, best_accs['clu_acc'])
                # record the best accuracy of K-means clustering
                log = open(os.path.join(args.log, 'log.txt'), 'a')
                log.write('\n                                                          best cluster acc: %3f' % best_accs['clu_acc'])
                if args.num_neurons:
                    cen_2 = c_set['s_2'] if cs_flag else c_set['t_2']
                    best_accs['clu_acc_2'], c_set['t_2'] = CLUSTER_GETTERS[args.cluster_method](f_set['t_2'], l_set['t'], l_set['t_p'] if args.cluster_method == 'kernel_kmeans' else cen_2, train_loader_target, epoch, model, args, best_accs['clu_acc_2'], change_target=False)
                    log.write('\n                                                          best cluster acc 2: %3f' % best_accs['clu_acc_2'])
                log.close()
                train_loader_target_batch = enumerate(train_loader_target)
            
            # re-initialize learnable cluster centers
            cen_set['1'].data = (c_set['t'] + c_set['s']).data / 2 if args.init_cen_on_st else c_set['t'].data
            if args.num_neurons:
                cen_set['2'].data = (c_set['t_2'] + c_set['s_2']).data / 2 if args.init_cen_on_st else c_set['t_2'].data
            
            # select source samples
            if itern and (args.src_soft_select or args.src_hard_select):
                src_cs = source_select(f_set['s'], l_set['s'], f_set['t'], l_set['t_p'], train_loader_source, epoch, c_set['t'], args) # c_set['t'] is used, consider clustering or not
                if args.src_hard_select:
                    train_loader_source_batch = enumerate(train_loader_source)
            
            # use source pre-trained model to extract features for first clustering and then recover initial model state
            if not itern and args.src_pretr_first: 
                model.load_state_dict(init_state_dict)
                
            if itern != start_itern:
                epoch += 1
                scheduler.step()
            
            del c_set, f_set, l_set
            gc.collect()
            torch.cuda.empty_cache()
        elif itern % test_freq == 0:
            accs = validate(val_loader_target, model, criterion, epoch, args, flag='')
            if args.tar != args.tar_t:
                accs.update(validate(val_loader_target_t, model, criterion, epoch, args))
            test_flag = True
        if test_flag:
            # record the best prec1 and save checkpoint
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            for k in accs.keys():
                is_best = accs[k] > best_accs[k]
                if is_best:
                    best_accs[k] = accs[k]
                    log = open(os.path.join(args.log, 'log.txt'), 'a')
                    log.write('\n                                                                         best ' + k + ': %3f' % (best_accs[k]))
                    log.close()            
                
                # save checkpoint
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'model_state_dict': model.state_dict(),
                    'cen_set': cen_set,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_accs': best_accs,
                }, is_best, args, suffix='_' + k)
            
            # reset records
            for k in train_records.keys():
                train_records[k].reset()
                
            test_flag = False
        
        # early stop
        if epoch > args.stop_epoch:
            break

        # train for one iteration
        train_loader_source_batch, train_loader_target_batch = train(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model, cen_set, criterion_cons, optimizer, itern, epoch, src_cs, args, train_records)
        model = model.cuda()
    
    # end time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    for k in best_accs.keys():
        log.write('\n * best ' + k + ': %3f' % best_accs[k])
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------\n')
    log.close()


def count_epoch_on_large_dataset(train_loader_target, train_loader_source, args):
    batch_number_t = len(train_loader_target)
    batch_number = batch_number_t
    if args.src_cls:
        batch_number_s = len(train_loader_source)
        if batch_number_s > batch_number_t:
            batch_number = batch_number_s
    
    return batch_number
    

def save_checkpoint(state, is_best, args, filename='checkpoint', prefix='', suffix=''):
    save_file = os.path.join(args.log, '{}.pth.tar'.format(prefix + filename))
    torch.save(state, save_file)
    if is_best:
        shutil.copyfile(save_file, os.path.join(args.log, 'model_best{}.pth.tar'.format(suffix)))


if __name__ == '__main__':
    main()


