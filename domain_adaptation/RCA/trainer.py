import time
import torch
import os
import math
import copy
import ipdb
import torch.nn as nn
import numpy as np
import gc
from torch.autograd import Variable
import torch.nn.functional as F


def train(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model, criterion, criterion_em, criterion_vat, optimizer, itern, current_epoch, train_records, args, reverse_grad_layer_index):
    model.train()

    lam = 2 / (1 + math.exp(-1 * 10 * current_epoch / args.epochs)) - 1 # penalty parameter
    
    end = time.time()    
    # prepare data for model forward and backward
    try:
        (input_source, target_source) = train_loader_source_batch.__next__()[1]
    except StopIteration:
        train_loader_source_batch = enumerate(train_loader_source)
        (input_source, target_source) = train_loader_source_batch.__next__()[1]
    target_source = target_source.cuda(non_blocking=True)
    input_source_var = Variable(input_source)
    target_source_var = Variable(target_source)
    target_source_var2 = Variable(target_source + args.num_classes)

    try:
        data = train_loader_target_batch.__next__()[1]
    except StopIteration:
        train_loader_target_batch = enumerate(train_loader_target)
        data = train_loader_target_batch.__next__()[1]
    
    input_target = data[0]
    input_target_var = Variable(input_target)

    train_records['data_time'].update(time.time() - end)    
    # calculate for the source data #######################################################
    output_s1, output_s2 = model(input_source_var)
    output_t1, output_t2 = model(input_target_var)
    target_target = output_t1.max(1)[1]
    target_target_var = Variable(target_target)
    target_target_var2 = Variable(target_target + args.num_classes)
    
    loss_min = criterion(output_s1, target_source_var) + lam * (criterion(output_s2, target_source_var) + criterion(output_t2, target_target_var2))
    loss_max = criterion(output_s1, target_source_var) + lam * (criterion(output_s2, target_source_var2) + criterion(output_t2, target_target_var))
    
    if args.em:
        loss_min = loss_min + 0.1 * (- torch.mean(torch.log(torch.mean(F.softmax(output_t1, dim=-1), 0) + 1e-6)))#criterion_em(output_t1)
        loss_max = loss_max + lam * criterion_em(output_t1)
    
    if args.vat:
        loss_min = loss_min + 1.0 * (criterion_vat(input_source_var, output_s1) + criterion_vat(input_target_var, output_t1))
        loss_max = loss_max + 1.0 * (criterion_vat(input_source_var, output_s1) + criterion_vat(input_target_var, output_t1))
    
    # record losses and accuracies on source data
    train_records['losses_min'].update(loss_min.item(), input_source.size(0))
    train_records['losses_max'].update(loss_max.item(), input_source.size(0))
    prec1 = accuracy(output_s1, target_source, topk=(1,))[0]
    train_records['top1_c'].update(prec1.item(), input_source.size(0))

    model.zero_grad()
    '''
    loss_min.backward(retain_graph=True)
    temp_grad = []
    for param in model.parameters():
        temp_grad.append(param.grad.data.clone())
    grad_for_classifier = temp_grad
    
    model.zero_grad()
    loss_max.backward()
    temp_grad = []
    for param in model.parameters():
        temp_grad.append(param.grad.data.clone())
    grad_for_featureExtractor = temp_grad
    
    count = 0
    for param in model.parameters():
        temp_grad = param.grad.data.clone()
        temp_grad.zero_()
        if count < reverse_grad_layer_index:
            temp_grad = temp_grad + grad_for_featureExtractor[count]
        else:
            temp_grad = temp_grad + grad_for_classifier[count]
        temp_grad = temp_grad
        param.grad.data = temp_grad
        count = count + 1
    '''
    for param in model.named_parameters():
        if param[0].find('fc') == -1:
            param[1].requires_grad = False
    loss_min.backward(retain_graph=True)
    
    for param in model.named_parameters():
        if param[0].find('fc') != -1:
            param[1].requires_grad = False
        else:
            param[1].requires_grad = True
    loss_max.backward()
    
    optimizer.step()
    
    model.zero_grad()    
    for param in model.named_parameters():
        if param[0].find('fc') != -1:
            param[1].requires_grad = True

    train_records['batch_time'].update(time.time() - end)
    if (itern + 1) % args.print_freq == 0:
        display = 'Train - epoch [{0}/{1}]({2})'.format(current_epoch, args.epochs, itern)
        for k in train_records.keys():
            display += '\t' + k + ': {ph.avg:.3f}'.format(ph=train_records[k])
        print(display)

        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\n' + display.replace('\t', ', '))
        log.close()
    
    return train_loader_source_batch, train_loader_target_batch


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mcp = MeanClassPrecision(args.num_classes)
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        with torch.no_grad():
            output, _ = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        mcp.update(output.data, target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('Test: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    print(str(mcp))
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\n                                    Test - epoch: %d, loss: %4f, Top1 acc: %3f, Top5 acc: %3f" %\
              (epoch, losses.avg, top1.avg, top5.avg))
    log.write('\n                                           ' + str(mcp))
    log.close()
    
    accs = {'prec1': top1.avg, 'mcp': mcp.mean_class_prec}
    
    return accs


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class MeanClassPrecision(object):
    """Computes and stores the mean class precision"""
    def __init__(self, num_classes, fmt=':.3f'):
        self.num_classes = num_classes
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.total_vector = torch.zeros(self.num_classes)
        self.correct_vector = torch.zeros(self.num_classes)
        self.per_class_prec = torch.zeros(self.num_classes)
        self.mean_class_prec = 0

    def update(self, output, target):
        pred = output.max(1)[1]
        correct = pred.eq(target).float().cpu()
        for i in range(target.size(0)):
            self.total_vector[target[i]] += 1
            self.correct_vector[target[i]] += correct[i]
        temp = torch.zeros(self.total_vector.size())
        temp[self.total_vector == 0] = 1e-6
        self.per_class_prec = self.correct_vector / (self.total_vector + temp) * 100
        self.mean_class_prec = self.per_class_prec.mean().item()
    
    def __str__(self):
        fmtstr = 'per-class prec: ' + '|'.join([str(i) for i in list(np.around(np.array(self.per_class_prec), int(self.fmt[-2])))])
        fmtstr = 'Mean class prec: {mean_class_prec' + self.fmt + '}, ' + fmtstr
        return fmtstr.format(**self.__dict__)
    

