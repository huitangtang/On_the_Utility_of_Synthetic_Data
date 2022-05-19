import time
import torch
import os
import math
import ipdb
import numpy as np
import torch.nn.functional as F

def train(source_train_loader, source_train_loader_batch, target_train_loader, target_train_loader_batch, model_set, criterion_set, optimizer_set, itern, current_epoch, train_records, args):
    model_set['G'].train()
    model_set['F1'].train()
    model_set['F2'].train()
    
    lam = 2 / (1 + math.exp(-1 * 10 * current_epoch / args.epochs)) - 1 # penalty parameter
    
    end = time.time()    
    # prepare data for model forward and backward
    try:
        (input_source, target_source) = source_train_loader_batch.__next__()[1]
    except StopIteration:
        source_train_loader_batch = enumerate(source_train_loader)
        (input_source, target_source) = source_train_loader_batch.__next__()[1]
    target_source = target_source.cuda(non_blocking=True)

    try:
        data = target_train_loader_batch.__next__()[1]
    except StopIteration:
        target_train_loader_batch = enumerate(target_train_loader)
        data = target_train_loader_batch.__next__()[1]
    
    input_target = data[0]

    train_records['data_time'].update(time.time() - end)

    if not args.concat_st:
        ####################################################### our way to calculate the output
        feature_source = model_set['G'](input_source)
        logit1_source = model_set['F1'](feature_source)
        logit2_source = model_set['F2'](feature_source)

        feature_target = model_set['G'](input_target)
        logit1_target = model_set['F1'](feature_target)
        logit2_target = model_set['F2'](feature_target)
    else:
        ####################################################### MCD way to calculate the output
        concat_input_st = torch.cat((input_source, input_target), 0)
        concat_feature_st = model_set['G'](concat_input_st)
        concat_logit1_st = model_set['F1'](concat_feature_st)
        concat_logit2_st = model_set['F2'](concat_feature_st)

        logit1_source = concat_logit1_st[:args.batch_size_s, :]
        logit2_source = concat_logit2_st[:args.batch_size_s, :]
        logit1_target = concat_logit1_st[args.batch_size_s:, :]
        logit2_target = concat_logit2_st[args.batch_size_s:, :]
    ###################################################### Over: two way to calculate the output
    prob1_target = F.softmax(logit1_target, dim=1)
    prob2_target = F.softmax(logit2_target, dim=1)
    ######################################################
    ce1_source_loss = criterion_set['ce'](logit1_source, target_source)
    ce2_source_loss = criterion_set['ce'](logit2_source, target_source)

    entropy1_target_loss = - torch.mean(torch.log(torch.mean(prob1_target, 0) + 1e-6))
    entropy2_target_loss = - torch.mean(torch.log(torch.mean(prob2_target, 0) + 1e-6))
    #### above: their entropy loss; below: our entropy loss
    our_entropy1_target_loss = criterion_set['em'](logit1_target)
    our_entropy2_target_loss = criterion_set['em'](logit2_target)

    if args.setting == 'mcd':
        all_loss = ce1_source_loss + ce2_source_loss
    elif args.setting == 'mcd_their_entropy':
        all_loss = ce1_source_loss + ce2_source_loss + 0.01 * (entropy1_target_loss + entropy2_target_loss)
    elif args.setting == 'mcd_our_entropy':
        all_loss = ce1_source_loss + ce2_source_loss + 0.01 * (our_entropy1_target_loss + our_entropy2_target_loss)
    else:
        raise ValueError('Unrecognized setting:', args.setting)
    #################################################################
    optimizer_set['G'].zero_grad()
    optimizer_set['F'].zero_grad()
    all_loss.backward()
    optimizer_set['G'].step()
    optimizer_set['F'].step()

    # mesure accuracy and record loss
    prec1_1 = accuracy(logit1_source, target_source, topk=(1,))[0]
    prec1_2 = accuracy(logit2_source, target_source, topk=(1,))[0]
    train_records['losses_s1'].update(all_loss.item(), input_source.size(0))
    train_records['top1_c1'].update(prec1_1.item(), input_source.size(0))
    train_records['top1_c2'].update(prec1_2.item(), input_source.size(0))

    ####################################################################
    ### step B: train classifier to maximize discrepancy. ##############
    ####################################################################
    eta = 1.0
    optimizer_set['G'].zero_grad()
    optimizer_set['F'].zero_grad()
    if not args.concat_st:
        ####################################################### our way to calculate the output
        feature_source = model_set['G'](input_source)
        logit1_source = model_set['F1'](feature_source)
        logit2_source = model_set['F2'](feature_source)

        feature_target = model_set['G'](input_target)
        logit1_target = model_set['F1'](feature_target)
        logit2_target = model_set['F2'](feature_target)
    else:
        ####################################################### MCD way to calculate the output
        concat_input_st = torch.cat((input_source, input_target), 0)
        concat_feature_st = model_set['G'](concat_input_st)
        concat_logit1_st = model_set['F1'](concat_feature_st)
        concat_logit2_st = model_set['F2'](concat_feature_st)

        logit1_source = concat_logit1_st[:args.batch_size_s, :]
        logit2_source = concat_logit2_st[:args.batch_size_s, :]
        logit1_target = concat_logit1_st[args.batch_size_s:, :]
        logit2_target = concat_logit2_st[args.batch_size_s:, :]
    ###################################################### Over: two way to calculate the output
    prob1_target = F.softmax(logit1_target, dim=1)
    prob2_target = F.softmax(logit2_target, dim=1)
    ######################################################
    ce1_source_loss = criterion_set['ce'](logit1_source, target_source)
    ce2_source_loss = criterion_set['ce'](logit2_source, target_source)
    entropy1_target_loss = - torch.mean(torch.log(torch.mean(prob1_target, 0) + 1e-6))
    entropy2_target_loss = - torch.mean(torch.log(torch.mean(prob2_target, 0) + 1e-6))
    #### above: their entropy loss; below: our entropy loss
    our_entropy1_target_loss = criterion_set['em'](logit1_target)
    our_entropy2_target_loss = criterion_set['em'](logit2_target)
    dis_target_loss = torch.mean(torch.abs(prob1_target - prob2_target))

    if args.setting == 'mcd':
        F_loss = ce1_source_loss + ce2_source_loss - eta * dis_target_loss
    elif args.setting == 'mcd_their_entropy':
        F_loss = ce1_source_loss + ce2_source_loss - eta * dis_target_loss + 0.01 * (entropy1_target_loss + entropy2_target_loss)
    elif args.setting == 'mcd_our_entropy':
        F_loss = ce1_source_loss + ce2_source_loss - eta * dis_target_loss + 0.01 * (our_entropy1_target_loss + our_entropy2_target_loss)
    else:
        raise ValueError('unrecognized setting:', args.setting)
    F_loss.backward()
    optimizer_set['F'].step()
    
    # record loss
    train_records['losses_s2'].update(F_loss.item(), input_source.size(0))

    ################################################################################
    ########################## Step C: train generator to minimize discrepancy #####
    ################################################################################
    for k in range(args.num_k):
        optimizer_set['G'].zero_grad()
        optimizer_set['F'].zero_grad()
        
        feature_target = model_set['G'](input_target)
        logit1_target = model_set['F1'](feature_target)
        logit2_target = model_set['F2'](feature_target)
        ###################################################### Over: two way to calculate the output
        prob1_target = F.softmax(logit1_target, dim=1)
        prob2_target = F.softmax(logit2_target, dim=1)
        G_loss = torch.mean(torch.abs(prob1_target - prob2_target))
        G_loss.backward()
        optimizer_set['G'].step()
    
    # record loss
    train_records['losses_s3'].update(G_loss.item(), input_target.size(0))


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


def validate(val_loader, model_set, epoch, args):
    batch_time = AverageMeter()
    top1_1 = AverageMeter()
    top1_2 = AverageMeter()
    mcp_1 = MeanClassPrecision(args.num_classes)
    mcp_2 = MeanClassPrecision(args.num_classes)
    # switch to evaluate mode
    model_set['G'].eval()
    model_set['F1'].eval()
    model_set['F2'].eval()
    end = time.time()
    for i, (input, target, _) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        # compute output
        with torch.no_grad():
            feature = model_set['G'](input)
            output1 = model_set['F1'](feature)
            output2 = model_set['F2'](feature)
        # measure accuracy
        prec1_1 = accuracy(output1, target, topk=(1,))[0]
        prec1_2 = accuracy(output2, target, topk=(1,))[0]
        top1_1.update(prec1_1.item(), input.size(0))
        top1_2.update(prec1_2.item(), input.size(0))
        mcp_1.update(output1, target)
        mcp_2.update(output2, target)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Test: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'top1@c1 {top1_1.val:.3f} ({top1_1.avg:.3f})\t'
                  'top1@c2 {top1_2.val:.3f} ({top1_2.avg:.3f})'.format(
                   epoch, i, len(val_loader), batch_time=batch_time,
                   top1_1=top1_1, top1_2=top1_2))

    print(' * Top1@c1 {top1_1.avg:.3f} Top1@c2 {top1_2.avg:.3f}'.format(top1_1=top1_1, top1_2=top1_2))
    print(str(mcp_1))
    print(str(mcp_2))
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\n                                    Test - epoch: %d, Top1-c1: %3f, Top1-c2: %3f" %\
              (epoch, top1_1.avg, top1_2.avg))
    log.write('\n                                           ' + str(mcp_1))
    log.write('\n                                           ' + str(mcp_2))
    log.close()

    accs = {'prec1': top1_1.avg, 'mcp': mcp_1.mean_class_prec}
    
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
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
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
    

