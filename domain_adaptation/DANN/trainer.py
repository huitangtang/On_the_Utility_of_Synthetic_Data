import time
import torch
from torch.autograd import Variable
import os
import math
import ipdb
import numpy as np


def train(source_train_loader, source_train_loader_batch, target_train_loader, target_train_loader_batch, model, criterion, optimizer, itern, current_epoch, train_records, args):
    model.train() # turn to training mode
    
    lam = 2 / (1 + math.exp(-1 * 10 * current_epoch / args.epochs)) - 1 # penalty parameter
    
    end = time.time()    
    # prepare data for model forward and backward
    try:
        (input_source, target_source) = source_train_loader_batch.__next__()[1]
    except StopIteration:
        source_train_loader_batch = enumerate(source_train_loader)
        (input_source, target_source) = source_train_loader_batch.__next__()[1]
    target_source = target_source.cuda(non_blocking=True)
    input_source_var = Variable(input_source)
    target_source_var = Variable(target_source)
    target_source_domain = torch.LongTensor(input_source.size(0)).fill_(0).cuda(non_blocking=True)
    target_source_domain_var = Variable(target_source_domain)

    try:
        data = target_train_loader_batch.__next__()[1]
    except StopIteration:
        target_train_loader_batch = enumerate(target_train_loader)
        data = target_train_loader_batch.__next__()[1]
    
    input_target = data[0]
    input_target_var = Variable(input_target)
    target_target_domain = torch.LongTensor(input_target.size(0)).fill_(1).cuda(non_blocking=True)
    target_target_domain_var = Variable(target_target_domain)    

    train_records['data_time'].update(time.time() - end)    
    
    output_source_lp, output_source_dc = model(input_source_var, lam)
    loss_source_lp = criterion(output_source_lp, target_source_var)
    loss_source_dc = criterion(output_source_dc, target_source_domain_var)    

    _, output_target_dc = model(input_target_var, lam)
    loss_target_dc = criterion(output_target_dc, target_target_domain_var)
    
    loss_dc = loss_source_dc + loss_target_dc    
    loss_overall = loss_source_lp + loss_dc
    
    # record losses and accuracies on source data
    train_records['losses_overall'].update(loss_overall.item(), input_source.size(0))
    train_records['losses_lp'].update(loss_source_lp.item(), input_source.size(0))
    train_records['losses_dc'].update(loss_source_dc.item(), input_source.size(0))
    prec1 = accuracy(output_source_lp, target_source, topk=(1,))[0]
    train_records['top1_lp'].update(prec1.item(), input_source.size(0))
    prec1 = accuracy(torch.cat((output_source_dc, output_target_dc), 0), torch.cat((target_source_domain, target_target_domain), 0), topk=(1,))[0]
    train_records['top1_dc'].update(prec1.item(), input_source.size(0) + input_target.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss_overall.backward()
    optimizer.step()
    
    train_records['batch_time'].update(time.time() - end)
    if (itern + 1) % args.print_freq == 0:
        display = 'Train - epoch [{0}/{1}]({2})'.format(current_epoch, args.epochs, itern)
        for k in train_records.keys():
            display += '\t' + k + ': {ph.avg:.3f}'.format(ph=train_records[k])
        print(display)

        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\n' + display.replace('\t', ', '))
        log.close()
    
    return source_train_loader_batch, target_train_loader_batch


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
            output, _ = model(input_var, 1)
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
    

