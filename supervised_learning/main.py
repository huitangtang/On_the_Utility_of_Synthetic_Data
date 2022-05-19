import argparse
import os
import random
import shutil
import time
import warnings
import logging
import numpy as np
from PIL import ImageFilter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import models.vision_transformer as vits
from models.mlp_mixer import MLPMixer
from models.vit import ViT
from models.resnet import ResBase

import ipdb
import gc


mean = {'imagenet': [0.485, 0.456, 0.406], 'subset_visda': [0.8781, 0.8769, 0.8743], 'subset_syn_id_s1': [0.4890, 0.4486, 0.4644]}
std = {'imagenet': [0.229, 0.224, 0.225], 'subset_visda': [0.2075, 0.2105, 0.2168], 'subset_syn_id_s1': [0.2386, 0.2528, 0.2712]}


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--fix_dataset', action='store_true', 
                    help = 'whether to fix dataset')
parser.add_argument('--data_size', type=str, default='all', choices=['all', 'part'], 
                    help = 'whether using all generated data till now for training or just the unseen part')
parser.add_argument('--data_train', type=str, metavar='DIR', default='/RenderImgs/subset_visda/train/',
                    help='path to train dataset')
parser.add_argument('--data_eval', type=str, metavar='DIR', nargs='+',
                    help='path to eval SYN_ID, SYN_OOD, and REAL datasets')
parser.add_argument('--exp_name', type=str, metavar='DIR', default=None,
                    help='experimental name')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names + ['vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small', 
                    'ViT', 'ViT_S', 'MLPMixer_S', 'MLPMixer', 'MLPMixer_L', 'MLPMixer_H'], help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--iters', default=None, type=int, metavar='N',
                    help='number of total iters to run')
parser.add_argument('--warmup_iters', default=0, type=int, metavar='N',
                    help='number of warmup iters to run')
parser.add_argument('--iters_lr', default=None, type=int, metavar='N',
                    help='number of iters to change lr for step decay')
parser.add_argument('--start-iter', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--decay_type', default='cosine', type=str, choices=['cosine', 'linear'],
                    help='type of learning rate decay')
parser.add_argument('--optimizer_type', default='sgd', type=str, choices=['sgd', 'adam'],
                    help='type of optimizer')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--evaluate_freq', default=500, type=int,
                    metavar='N', help='evaluate frequency (default: 500)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--data_augmentation', default='strong', type=str, choices=['none', 'weak', 'strong'],
                    help='which kind of data augmentation to use during training')
parser.add_argument('--norm_stat', default='imagenet', type=str, choices=['imagenet', 'subset_visda', 'subset_syn_id_s1'],
                    help='on which dataset the normalization statistics is computed')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num_classes', default=10, type=int,
                    help='number of classes for training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--patch_size', default=16, type=int, help='patch_size for viT')


best_acc1s = {}
best_mcps = {}
Total_lr_iters = None
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = './checkpoints/'
LOG_PATH = './logs/'
EVENT_PATH = './events/'

for path in [CHECKPOINT_PATH, LOG_PATH, EVENT_PATH]:
    if not os.path.exists(path):
            os.makedirs(path)

def main():
    global Total_lr_iters
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')
    
    Total_lr_iters = args.iters# // 2
    
    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    global best_acc1s, best_mcps
    data_eval = {}
    if args.data_eval:
        data_eval_type = ['syn_id', 'syn_ood', 'real', 'syn_id2', 'syn_ood2']
        data_eval = {data_eval_type[i]: args.data_eval[i] for i in range(len(args.data_eval))}
    for k in data_eval.keys():
        best_acc1s[k] = 0 # initialize best_acc1 as 0
        best_mcps[k] = 0 # initialize best_mcp as 0

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating new model '{}'".format(args.arch))
    if args.arch.lower().startswith('resnet'):
        model = ResBase(option=args.arch, pret=args.pretrained, num_classes=args.num_classes)
    elif args.arch.find('MLPMixer') != -1:
        kwargs = {'image_size': 224, 'patch_size': 16, 'dim': 768, 'depth': 12, 'num_classes': args.num_classes}
        if args.arch == 'MLPMixer_S':
            kwargs['dim'] = 512
            kwargs['depth'] = 8
        elif args.arch == 'MLPMixer_L':
            kwargs['dim'] = 1024
            kwargs['depth'] = 24
        elif args.arch == 'MLPMixer_H':
            kwargs['patch_size'] = 14
            kwargs['dim'] = 1280
            kwargs['depth'] = 32
        model = MLPMixer(**kwargs)
    elif args.arch == 'ViT':
        model = ViT(
                image_size = 224,
                patch_size = 16,
                dim = 768,
                depth = 12,
                heads = 12,
                mlp_dim = 768*4,
                dropout = 0.1,
                emb_dropout = 0.1,
                num_classes = args.num_classes
            )
    elif args.arch == 'ViT_S':
        model = ViT(
                image_size = 224,
                patch_size = 16,
                dim = 384,
                depth = 12,
                heads = 6,
                mlp_dim = 384*4,
                dropout = 0.1,
                emb_dropout = 0.1,
                num_classes = args.num_classes
            )
    elif args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=0.1,  # stochastic depth
            num_classes = args.num_classes
        )
    else:
        model = models.__dict__[args.arch](num_classes=args.num_classes) # pretrained=False

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    iter_part_resume = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_iter = checkpoint['iter']
            iter_part_resume = checkpoint['iter_part_resume']
            best_acc1s = checkpoint['best_acc1s']
            try:
                best_mcps = checkpoint['best_mcps']
            except:
                pass
            try:
                model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
            except:
                try:
                    model.load_state_dict({'module.'+k: v for k, v in checkpoint['state_dict'].items()})
                except:
                    model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (iter {})"
                  .format(args.resume, checkpoint['iter']))
            del checkpoint
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit()

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=mean[args.norm_stat], std=std[args.norm_stat])
    
    val_loaders = {k: torch.utils.data.DataLoader(
        datasets.ImageFolder(v, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True) for k, v in data_eval.items()}
    
    if args.evaluate:
        args.filename = 'eval'
        logging.basicConfig(filename=os.path.join(LOG_PATH, '{}.txt'.format(args.filename)), format="%(asctime)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
        logger.setLevel(logging.INFO)
        for k, v in val_loaders.items():
            logger.info('Only Test!\n =checkpoint: ' + args.resume + ' ({})'.format(checkpoint['iter']) + '\n =eval_folder: ' + data_eval[k])
            validate(v, model, criterion, args)
            logger.info('Tested!\n')
        return
    
    if args.data_augmentation == 'strong':
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                flip_and_color_jitter,
                GaussianBlur(),
                transforms.ToTensor(),
                normalize,
        ])
    elif args.data_augmentation == 'weak':
        image_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize,
        ])
    else:
        image_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
        ])
    train_dataset = datasets.ImageFolder(args.data_train, image_transforms)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    
    ## training begin
    len_train_loader = len(train_loader)
    print('train_loader length: ', len_train_loader)
    iter_train_loader = iter(train_loader)
    iter_resume = args.start_iter if args.data_size == 'all' else iter_part_resume
    if args.resume and not args.fix_dataset: # we must use a fixed seed to resume the dataloader!!!
        for i in range(iter_resume):
            if i % 100 == 0:
                print(i)
            next(iter_train_loader)

    ## filename
    num_imgs = sum([len(x) for _, _, x in os.walk(args.data_train)])
    args.filename = 'fix_dataset_{}-{}imgs-{}-{}iters-lr{}-{}iters_lr-batchsize{}-aug_{}-s{}'.format(args.fix_dataset, num_imgs, args.arch, args.iters, 
                    args.lr, args.iters_lr, args.batch_size, args.data_augmentation, args.seed)
    if args.exp_name is not None:
        args.filename = args.exp_name + '-' + args.filename
    print('Saving file with filename: ', args.filename)
    writers = {k: SummaryWriter(os.path.join(EVENT_PATH, args.filename, k)) for k in data_eval.keys()}
    logging.basicConfig(filename=os.path.join(LOG_PATH, '{}.txt'.format(args.filename)), format="%(asctime)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    logger.setLevel(logging.INFO)
    logger.info(dict(args._get_kwargs()))
    logger.info('Image folder --- ' + args.data_train + ' --- run')
    logger.info("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    
    batch_time = AverageMeter('Time')
    data_time = AverageMeter('Data')
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    end = time.time()
    scheduler = lr_scheduler(base_lr=args.lr, final_lr=0, total_iters=Total_lr_iters, warmup_iters=args.warmup_iters, decay_type=args.decay_type)#args.lr*0.01
    if args.iters_lr is None:
        print('Using {:s} learning rate scheduler: '.format(args.decay_type), scheduler)
    elif args.iters_lr != 0:
        print('Using step (0.97 **) learning rate scheduler: ', args.iters_lr)
    else:
        print('Using unchange learning rate scheduler: ', args.lr)
    progress = ProgressMeter(args.iters, [batch_time, data_time, losses, top1, top5])
    for iter_ in range(args.start_iter, args.iters):
        if args.iters_lr is None:
            if iter_ < Total_lr_iters:
                lr = scheduler[iter_]
            else:
                lr = scheduler[-1]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif args.iters_lr != 0:
            adjust_learning_rate(optimizer, iter_, args)
        # switch to train mode
        model.train()
        
        num_past_batches = iter_resume + iter_ - args.start_iter
        if num_past_batches == len_train_loader and not args.fix_dataset:
            break
            
        if iter_ % len_train_loader == 0 and args.fix_dataset:
            iter_train_loader = iter(train_loader)
        images, target = next(iter_train_loader)
        # measure data loading time
        data_time.update(time.time() - end)
 
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iter_ % args.print_freq == 0:
            progress.display(iter_, prefix="Train: ")
            
        if ((iter_ + 1) % args.evaluate_freq == 0) or (iter_ == args.iters - 1):
            print(' # Train loss {:.3f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(losses.avg, top1=top1, top5=top5))
            
            logger.info('- Iter: [{}]'.format(iter_))
            logger.info('# Train loss {:.3f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                          .format(losses.avg, top1=top1, top5=top5))
            
            if not val_loaders:
                save_checkpoint({
                    'iter': iter_ + 1,
                    'iter_part_resume': 0 if args.data_size == 'all' or num_past_batches + 1 == len_train_loader else num_past_batches + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1s': best_acc1s,
                    'best_mcps': best_mcps,
                    'optimizer' : optimizer.state_dict(),
                }, False, args.filename)
            
            # evaluate on validation set
            for k, v in val_loaders.items():
                test_loss, test_acc1, test_mcp = validate(v, model, criterion, args)

                # remember best acc@1 and save checkpoint
                is_best = test_acc1 > best_acc1s[k]
                best_acc1s[k] = max(test_acc1, best_acc1s[k])
    
                save_checkpoint({
                    'iter': iter_ + 1,
                    'iter_part_resume': 0 if args.data_size == 'all' or num_past_batches + 1 == len_train_loader else num_past_batches + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1s': best_acc1s,
                    'best_mcps': best_mcps,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, args.filename, suffix='-'+k)
                
                is_best = test_mcp > best_mcps[k]
                best_mcps[k] = max(test_mcp, best_mcps[k])
    
                save_checkpoint({
                    'iter': iter_ + 1,
                    'iter_part_resume': 0 if args.data_size == 'all' or num_past_batches + 1 == len_train_loader else num_past_batches + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1s': best_acc1s,
                    'best_mcps': best_mcps,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, args.filename, suffix='-'+k+'-mcp')
                
                writers[k].add_scalar('train/1.train_loss', losses.avg, iter_)
                writers[k].add_scalar('train/2.train_acc@1', top1.avg, iter_)
                writers[k].add_scalar('test/1.test_loss', test_loss, iter_)
                writers[k].add_scalar('test/2.test_acc@1', test_acc1, iter_)
                writers[k].add_scalar('test/3.test_mcp', test_mcp, iter_)
                
                logger.info('* Best acc1 {:s}: {:.3f}'.format(k, best_acc1s[k]))
                logger.info('* Best mcp {:s}: {:.3f}\n'.format(k, best_mcps[k]))

            losses.reset()
            top1.reset()
            top5.reset()
            data_time.reset()
            batch_time.reset()
            
        if num_past_batches + 1 == len_train_loader or (iter_ + 1) in [2e1, 2e2, 2e3, 2e4, args.iters]:#% (args.iters // 5) == 0:
            save_checkpoint({
                'iter': iter_ + 1,
                'iter_part_resume': 0 if args.data_size == 'all' or num_past_batches + 1 == len_train_loader else num_past_batches + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1s': best_acc1s,
                'best_mcps': best_mcps,
                'optimizer' : optimizer.state_dict(),
            }, False, args.filename, prefix='{}iter-'.format(iter_))
    
    for k in writers.keys():
        writers[k].close()


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time')
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5])
    
    mcp = MeanClassPrecision(args.num_classes)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            mcp.update(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, prefix='Eval: ')

        # TODO: this should also be done with the ProgressMeter
        print(' * Eval loss {:.3f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(losses.avg, top1=top1, top5=top5))
        print(str(mcp))
        
        logger.info('* Eval loss {:.3f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(losses.avg, top1=top1, top5=top5))
        logger.info(str(mcp))

    return losses.avg, top1.avg, mcp.mean_class_prec

################################################################################
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def save_checkpoint(state, is_best, filename, prefix='', suffix=''):
    save_file = os.path.join(CHECKPOINT_PATH, '{}/{}.pth.tar'.format(filename, prefix + filename))
    save_dir = os.path.split(save_file)[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, save_file)
    if is_best:
        shutil.copyfile(save_file, os.path.join(save_dir, '{}-best.pth.tar'.format(filename + suffix)))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':.3f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters

    def display(self, batch, prefix=""):
        entries = [prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
        #logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, iter_, args):
    """Sets the learning rate to the initial LR decayed by * every * iters"""
    # lr = args.lr * (0.1 ** (iter_ // args.iters_lr))
    # lr = args.lr * (0.95 ** (iter_ // args.iters_lr))
    adjust_lr = (0.97 ** (iter_ // args.iters_lr))
    adjust_lr = max(adjust_lr, 0.01)
    lr = args.lr * adjust_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

def lr_scheduler(base_lr, final_lr, total_iters, warmup_iters=0, start_warmup_lr=0, decay_type='cosine'):
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_lr, base_lr, warmup_iters)

    iters = np.arange(total_iters - warmup_iters)
    if decay_type == 'cosine':
        schedule = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(np.pi * iters / len(iters)))
    elif decay_type == 'linear':
        schedule = final_lr + (base_lr - final_lr) * (1 - iters / len(iters))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_iters
    return schedule


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
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
        fmtstr = 'too many classes'
        if self.num_classes <= 20:
            fmtstr = 'per-class prec: ' + '|'.join([str(i) for i in list(np.around(np.array(self.per_class_prec), int(self.fmt[-2])))])
        fmtstr = 'Mean class prec: {mean_class_prec' + self.fmt + '}, ' + fmtstr
        return fmtstr.format(**self.__dict__)
    

if __name__ == '__main__':
    main()
