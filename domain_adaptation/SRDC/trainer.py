import time
import torch
import os
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils.kernel_kmeans import KernelKMeans
import gc
import ipdb

def train(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model, cen_set, criterion_cons, optimizer, itern, epoch, src_cs, args, train_records):  
    # switch to train mode
    model.train()

    lam = 2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1 # penalty parameter
    weight = lam if epoch and args.src_cls else 1

    end = time.time()
    
    # prepare target data
    try:
        data = train_loader_target_batch.__next__()[1]
    except StopIteration:
        train_loader_target_batch = enumerate(train_loader_target)
        data = train_loader_target_batch.__next__()[1]
    data = data[:-1]
    input_target = data[0]
    target_target = data[-1].cuda(non_blocking=True)

    train_records['data_time'].update(time.time() - end)
    
    # model forward on target
    f_t, f_t_2, ca_t = model(input_target.cuda())
    
    loss = 0
                
    loss += weight * TarDisClusterLoss(args, epoch, ca_t, target_target, em=(args.cluster_method == 'em'))
    
    if args.learn_embed:
        prob_pred = (1 + (f_t.unsqueeze(1) - cen_set['1'].unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
        loss += weight * TarDisClusterLoss(args, epoch, prob_pred, target_target, softmax=args.embed_softmax)
        if args.num_neurons and not args.no_second_embed:
            prob_pred_2 = (1 + (f_t_2.unsqueeze(1) - cen_set['2'].unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
            loss += weight * TarDisClusterLoss(args, epoch, prob_pred_2, target_target, softmax=args.embed_softmax)
    
    if args.aug_tar_agree:
        ca_t_dup = model(data[1].cuda())[-1]
        loss += weight * criterion_cons(ca_t, ca_t_dup)
        
    if args.src_cls:
        # prepare source data
        try:
            (input_source, target_source, index) = train_loader_source_batch.__next__()[1]
        except StopIteration:
            train_loader_source_batch = enumerate(train_loader_source)
            (input_source, target_source, index) = train_loader_source_batch.__next__()[1]
        target_source = target_source.cuda(non_blocking=True)
        
        # model forward on source
        f_s, f_s_2, ca_s = model(input_source.cuda())
        prec1 = accuracy(ca_s, target_source, topk=(1,))[0]
        train_records['top1_s'].update(prec1.item(), input_source.size(0))
        
        loss += SrcClassifyLoss(args, ca_s, target_source, index, src_cs, lam, fit=args.src_fit)
        
        if args.learn_embed:
            prob_pred = (1 + (f_s.unsqueeze(1) - cen_set['1'].unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
            loss += weight * SrcClassifyLoss(args, prob_pred, target_source, index, src_cs, lam, softmax=args.embed_softmax, fit=args.src_fit)
            if args.num_neurons and not args.no_second_embed:
                prob_pred_2 = (1 + (f_s_2.unsqueeze(1) - cen_set['2'].unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
                loss += weight * SrcClassifyLoss(args, prob_pred_2, target_source, index, src_cs, lam, softmax=args.embed_softmax, fit=args.src_fit)

    train_records['losses'].update(loss.item(), input_target.size(0))
    
    # loss backward and network update
    model.zero_grad()
    loss.backward()
    optimizer.step()

    train_records['batch_time'].update(time.time() - end)
    if (itern + 1) % args.print_freq == 0:
        display = 'Train - epoch [{0}/{1}]({2})'.format(epoch, args.epochs, itern)
        for k in train_records.keys():
            display += '\t' + k + ': {ph.avg:.3f}'.format(ph=train_records[k])
        print(display)

        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\n' + display.replace('\t', ', '))
        log.close()
    
    return train_loader_source_batch, train_loader_target_batch


def TarDisClusterLoss(args, epoch, output, target, softmax=True, em=False):
    if softmax:
        prob_p = F.softmax(output, dim=1)
    else:
        prob_p = output / output.sum(1, keepdim=True)
    if em:
        prob_q = prob_p
    else:
        prob_q1 = Variable(torch.cuda.FloatTensor(prob_p.size()).fill_(0))
        prob_q1.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1).cuda()) # assigned pseudo labels
        if (epoch == 0) or args.ao:
            prob_q = prob_q1
        else:
            prob_q2 = prob_p / prob_p.sum(0, keepdim=True).pow(0.5)
            prob_q2 /= prob_q2.sum(1, keepdim=True)
            prob_q = (1 - args.beta) * prob_q1 + args.beta * prob_q2
    
    if softmax:
        loss = - (prob_q * F.log_softmax(output, dim=1)).sum(1).mean()
    else:
        loss = - (prob_q * prob_p.log()).sum(1).mean()
    
    return loss
    
    
def SrcClassifyLoss(args, output, target, index, src_cs, lam, softmax=True, fit=False):
    if softmax:
        prob_p = F.softmax(output, dim=1)
    else:
        prob_p = output / output.sum(1, keepdim=True)
    prob_q = Variable(torch.cuda.FloatTensor(prob_p.size()).fill_(0))
    prob_q.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1).cuda())
    if fit:
        prob_q = (1 - prob_p) * prob_q + prob_p * prob_p    
    if args.src_mix_weight:
        src_weights = lam * src_cs[index] + (1 - lam) * torch.ones(output.size(0)).cuda()
    else:
        src_weights = src_cs[index]
    
    if softmax:
        loss = - (src_weights * (prob_q * F.log_softmax(output, dim=1)).sum(1)).mean()
    else:
        loss = - (src_weights * (prob_q * prob_p.log()).sum(1)).mean()
    
    return loss


def validate(val_loader, model, criterion, epoch, args, flag='_te'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    
    mcp = MeanClassPrecision(args.num_classes)
    
    end = time.time()
    for i, (input, target, _) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model(input.cuda())[-1]
            loss = criterion(output, target)

        # compute and record accuracy and loss
        prec1 = accuracy(output, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))
        mcp.update(output, target) # compute class-wise accuracy
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Evaluate on target - [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                  .format(epoch, i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    print(' * Evaluate on target - prec@1: {top1.avg:.3f}'.format(top1=top1))
    print(str(mcp))
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\n             Evaluate on target - epoch: %d, loss: %.4f, acc: %.3f" % (epoch, losses.avg, top1.avg))
    log.write('\n                                           ' + str(mcp))
    
    accs = {'prec1'+flag: top1.avg, 'mcp'+flag: mcp.mean_class_prec}
    
    return accs

    
def validate_compute_cen(val_loader_target, val_loader_source, model, criterion, epoch, args, compute_cen=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # switch to evaluate mode
    model.eval()

    # compute source class centroids
    f_set, l_set, c_set = dict(), dict(), dict()
    if compute_cen:
        f_set['s'] = torch.cuda.FloatTensor(len(val_loader_source.dataset.imgs), model.module.feat1_dim).fill_(0)
        l_set['s'] = torch.cuda.LongTensor(len(val_loader_source.dataset.imgs)).fill_(0)
        c_set['s'] = torch.cuda.FloatTensor(args.num_classes, model.module.feat1_dim).fill_(0)
        count_s = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
        if args.num_neurons:
            f_set['s_2'] = torch.cuda.FloatTensor(len(val_loader_source.dataset.imgs), model.module.feat2_dim).fill_(0)
            c_set['s_2'] = torch.cuda.FloatTensor(args.num_classes, model.module.feat2_dim).fill_(0)
        for i, (input, target, index) in enumerate(val_loader_source): # the iterarion in the source dataset
            target = target.cuda(non_blocking=True)
            index = index.cuda()
            with torch.no_grad():
                feature, feature_2, output = model(input.cuda())
            f_set['s'][index] = feature.data.clone()
            l_set['s'][index] = target.clone()
            target_ = torch.cuda.FloatTensor(output.size()).fill_(0)
            target_.scatter_(1, target.unsqueeze(1), torch.ones(target_.size(0), 1).cuda())
            c_set['s'] += ((feature / feature.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * target_.unsqueeze(2)).sum(0) if args.cluster_method == 'spherical_kmeans' else (feature.unsqueeze(1) * target_.unsqueeze(2)).sum(0)
            if args.num_neurons:
                f_set['s_2'][index] = feature_2.data.clone()
                c_set['s_2'] += ((feature_2 / feature_2.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * target_.unsqueeze(2)).sum(0) if args.cluster_method == 'spherical_kmeans' else (feature_2.unsqueeze(1) * target_.unsqueeze(2)).sum(0)
            count_s += target_.sum(0).unsqueeze(1)
        c_set = {k: v / (1 if args.cluster_method == 'spherical_kmeans' else count_s) for k, v in c_set.items()} # finalize source class centroids
    
    f_set['t'] = torch.cuda.FloatTensor(len(val_loader_target.dataset.imgs), model.module.feat1_dim).fill_(0)
    l_set['t'] = torch.cuda.LongTensor(len(val_loader_target.dataset.imgs)).fill_(0)
    l_set['t_p'] = torch.cuda.FloatTensor(len(val_loader_target.dataset.imgs), args.num_classes).fill_(0)
    c_set['t'] = torch.cuda.FloatTensor(args.num_classes, model.module.feat1_dim).fill_(0)
    count_t = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
    if args.num_neurons:
        f_set['t_2'] = torch.cuda.FloatTensor(len(val_loader_target.dataset.imgs), model.module.feat2_dim).fill_(0)
        c_set['t_2'] = torch.cuda.FloatTensor(args.num_classes, model.module.feat2_dim).fill_(0)
    
    mcp = MeanClassPrecision(args.num_classes)
    end = time.time()
    for i, (input, target, index) in enumerate(val_loader_target): # the iterarion in the target dataset
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        index = index.cuda()
        
        with torch.no_grad():
            feature, feature_2, output = model(input.cuda())
        
        f_set['t'][index] = feature.data.clone() # index:a tensor 
        l_set['t'][index] = target.clone()
        l_set['t_p'][index] = output.data.clone()
            
        # compute target class centroids
        pred = output.data.max(1)[1]
        pred_ = torch.cuda.FloatTensor(output.size()).fill_(0)
        pred_.scatter_(1, pred.unsqueeze(1), torch.ones(pred_.size(0), 1).cuda())
        c_set['t'] += ((feature / feature.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * pred_.unsqueeze(2)).sum(0) if args.cluster_method == 'spherical_kmeans' else (feature.unsqueeze(1) * pred_.unsqueeze(2)).sum(0)
        count_t += pred_.sum(0).unsqueeze(1)
        if args.num_neurons:
            f_set['t_2'][index] = feature_2.data.clone()
            c_set['t_2'] += ((feature_2 / feature_2.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * pred_.unsqueeze(2)).sum(0) if args.cluster_method == 'spherical_kmeans' else (feature_2.unsqueeze(1) * pred_.unsqueeze(2)).sum(0)
                    
        # compute and record loss and accuracy
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))
        prec1 = accuracy(output, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))
        mcp.update(output, target) # compute class-wise accuracy        
        
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Evaluate on target - [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                  .format(epoch, i, len(val_loader_target), batch_time=batch_time, loss=losses, top1=top1))

    # compute global class centroids
    c_set = {k: v / (count_t + args.eps if k.startswith('t') and args.cluster_method != 'spherical_kmeans' else 1) for k, v in c_set.items()} # finalize target class centroids

    print(' * Evaluate on target - prec@1: {top1.avg:.3f}'.format(top1=top1))
    print(str(mcp))
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\n             Evaluate on target - epoch: %d, loss: %.4f, acc: %.3f" % (epoch, losses.avg, top1.avg))
    log.write('\n                                           ' + str(mcp))
    
    accs = {'prec1': top1.avg, 'mcp': mcp.mean_class_prec}
    
    return accs, c_set, f_set, l_set


def source_select(source_features, source_targets, target_features, pseudo_labels, train_loader_source, epoch, cen, args):
    # compute source weights
    src_cs = 0.5 * (1 + F.cosine_similarity(source_features, cen[source_targets]))
    
    # hard source sample selection
    if args.src_hard_select:
        num_select_src_each_class = torch.cuda.LongTensor(args.num_classes).fill_(0)
        tao = 1 / (1 + math.exp(- args.tao * (epoch + 1))) - 0.01
        delta = np.log(args.num_classes) / 10
        indexes = torch.arange(0, source_features.size(0))
        
        interval = 10000
        target_kernel_sim = torch.cuda.FloatTensor()
        while target_features.size(0) != 0:
            target_kernel_sim = torch.cat([target_kernel_sim, (1 + (target_features[:interval].unsqueeze(1) - cen.unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)], dim=0)
            target_features = target_features[interval:]
        assert target_kernel_sim.size(0) == pseudo_labels.size(0)
        
        if args.embed_softmax:
            target_kernel_sim = F.softmax(target_kernel_sim, dim=1)
        else:
            target_kernel_sim /= target_kernel_sim.sum(1, keepdim=True)
        _, pseudo_cat_dist = target_kernel_sim.max(dim=1)
        pseudo_labels_softmax = F.softmax(pseudo_labels, dim=1)
        _, pseudo_cat_std = pseudo_labels_softmax.max(dim=1)
        
        selected_indexes = []
        for c in range(args.num_classes):            
            temp1 = target_kernel_sim[pseudo_cat_dist == c].mean(dim=0)
            temp2 = pseudo_labels_softmax[pseudo_cat_std == c].mean(dim=0)
            temp1 = - (temp1 * ((temp1 + args.eps).log())).sum(0) # entropy 1
            temp2 = - (temp2 * ((temp2 + args.eps).log())).sum(0) # entropy 2
            if (temp1 > delta) and (temp2 > delta):
                tao -= 0.1
            elif (temp1 <= delta) and (temp2 <= delta):
                pass
            else:
                tao -= 0.05
            while True:
                num_select_src_each_class[c] = (src_cs[source_targets == c] >= tao).float().sum()
                if num_select_src_each_class[c] > 0: # at least 1
                    selected_indexes.extend(list(np.array(indexes[source_targets == c][src_cs[source_targets == c] >= tao])))
                    break
                else:
                    tao -= 0.05
        
        train_loader_source.dataset.samples = []
        train_loader_source.dataset.tgts = []
        for idx in selected_indexes:
            train_loader_source.dataset.samples.append(train_loader_source.dataset.imgs[idx])
            train_loader_source.dataset.tgts.append(train_loader_source.dataset.imgs[idx][1])
        print('%d source instances have been selected at %d epoch' % (len(selected_indexes), epoch))
        print('Number of selected source instances each class: ', num_select_src_each_class)
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\n~~~%d source instances have been selected at %d epoch~~~' % (len(selected_indexes), epoch))
        log.close()
        
        src_cs = torch.cuda.FloatTensor(len(train_loader_source.dataset.tgts)).fill_(1)
    
    del source_features, source_targets, pseudo_labels
    gc.collect()
    torch.cuda.empty_cache()
    
    return src_cs
    

def kernel_kmeans(target_features, target_targets, pseudo_labels, train_loader_target, epoch, model, args, best_prec, change_target=True):
    # define kernel k-means clustering
    kkm = KernelKMeans(n_clusters=args.num_classes, max_iter=args.cluster_iter, random_state=0, kernel=args.cluster_kernel, gamma=args.gamma, verbose=1)
    kkm.fit(np.array(target_features.cpu()), initial_label=np.array(pseudo_labels.max(1)[1].long().cpu()), true_label=np.array(target_targets.cpu()), args=args, epoch=epoch)
    
    idx_sim = torch.from_numpy(kkm.labels_)
    c_tar = torch.cuda.FloatTensor(args.num_classes, target_features.size(1)).fill_(0)
    count = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
    for i in range(target_targets.size(0)):
        c_tar[idx_sim[i]] += target_features[i]
        count[idx_sim[i]] += 1
        if change_target:
            train_loader_target.dataset.tgts[i] = idx_sim[i].item()
    c_tar /= (count + args.eps)
    
    prec1 = kkm.prec1_
    is_best = prec1 > best_prec
    if is_best:
        best_prec = prec1
        #torch.save(c_tar, os.path.join(args.log, 'c_t_kernel_kmeans_cluster_best.pth.tar'))
        #torch.save(model.state_dict(), os.path.join(args.log, 'checkpoint_kernel_kmeans_cluster_best.pth.tar'))
    
    del target_features, target_targets, pseudo_labels
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_prec, c_tar


def kmeans(target_features, target_targets, c, train_loader_target, epoch, model, args, best_prec, change_target=True):
    batch_time = AverageMeter()
    
    c_tar = c.data.clone()
    end = time.time()
    for itr in range(args.cluster_iter):
        interval = 2000
        dist_xt_ct = torch.cuda.FloatTensor(target_features.size(0), args.num_classes).fill_(0)
        for split in range(int(target_features.size(0) / interval) + 1):
            dist_xt_ct_temp = target_features[split*interval:(split+1)*interval].unsqueeze(1) - c_tar.unsqueeze(0)
            dist_xt_ct[split*interval:(split+1)*interval] = dist_xt_ct_temp.pow(2).sum(2)
            del dist_xt_ct_temp
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        _, idx_sim = (-1 * dist_xt_ct).topk(1, 1, True, True)
        prec1 = accuracy(-1 * dist_xt_ct, target_targets, topk=(1,))[0].item()
        is_best = prec1 > best_prec
        if is_best:
            best_prec = prec1
            #torch.save(c_tar, os.path.join(args.log, 'c_t_kmeans_cluster_best.pth.tar'))
            #torch.save(model.state_dict(), os.path.join(args.log, 'checkpoint_kmeans_cluster_best.pth.tar'))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        info = 'Epoch %d, k-means clustering %d, average clustering time %.3f, prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1)
        print(info)
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\n' + info)
        log.close()
        
        c_tar.fill_(0)
        count = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0) 
        for k in range(args.num_classes):
            c_tar[k] += target_features[idx_sim.squeeze(1) == k].sum(0)
            count[k] += (idx_sim.squeeze(1) == k).float().sum()
        c_tar /= (count + args.eps)
    
    if change_target:
        for i in range(target_targets.size(0)):
            train_loader_target.dataset.tgts[i] = int(idx_sim[i])
    
    del target_features, target_targets, dist_xt_ct
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_prec, c_tar

    
def spherical_kmeans(target_features, target_targets, c, train_loader_target, epoch, model, args, best_prec, change_target=True):
    batch_time = AverageMeter()
    
    c_tar = c.data.clone()
    end = time.time()
    for itr in range(args.cluster_iter):
        interval = 10000
        dist_xt_ct = torch.cuda.FloatTensor(target_features.size(0), args.num_classes).fill_(0)
        for split in range(int(target_features.size(0) / interval) + 1):
            dist_xt_ct_temp = target_features[split*interval:(split+1)*interval].unsqueeze(1) - c_tar.unsqueeze(0)
            dist_xt_ct[split*interval:(split+1)*interval] = dist_xt_ct_temp.pow(2).sum(2)
            del dist_xt_ct_temp
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        _, idx_sim = (-1 * dist_xt_ct).topk(1, 1, True, True)
        prec1 = accuracy(-1 * dist_xt_ct, target_targets, topk=(1,))[0].item()
        is_best = prec1 > best_prec
        if is_best:
            best_prec = prec1
            #torch.save(c_tar, os.path.join(args.log, 'c_t_spherical_kmeans_cluster_best.pth.tar'))
            #torch.save(model.state_dict(), os.path.join(args.log, 'checkpoint_spherical_kmeans_cluster_best.pth.tar'))
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        info = 'Epoch %d, spherical k-means clustering %d, average clustering time %.3f, prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1)
        print(info)
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\n' + info)
        log.close()
        c_tar.fill_(0)
        for k in range(args.num_classes):
            c_tar[k] += (target_features[idx_sim.squeeze(1) == k] / (target_features[idx_sim.squeeze(1) == k].norm(2, dim=1, keepdim=True) + args.eps)).sum(0)
        
    if change_target:
        for i in range(target_targets.size(0)):
            train_loader_target.dataset.tgts[i] = int(idx_sim[i])
    
    del target_features, target_targets, dist_xt_ct
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_prec, c_tar


CLUSTER_GETTERS = {'kernel_kmeans': kernel_kmeans, 
                  'kmeans': kmeans,
                  'spherical_kmeans': spherical_kmeans}


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
        

