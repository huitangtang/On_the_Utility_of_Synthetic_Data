import argparse


def opts():
    parser = argparse.ArgumentParser(description='Training DisClusterDA',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path_source', type=str, default='/data/',
                        help='root of the source dataset')
    parser.add_argument('--data_path_target_tr', type=str, default='/data/',
                        help='root of the target dataset (for training)')
    parser.add_argument('--data_path_target_te', type=str, default='/data/',
                        help='root of the target dataset (for test)')
    parser.add_argument('--src', type=str, default='', help='source domain')
    parser.add_argument('--tar_tr', type=str, default='', help='target domain (for training)')
    parser.add_argument('--tar_te', type=str, default='', help='target domain (for test)')
    parser.add_argument('--num_classes', type=int, default=31, help='class number')
    # general optimization options
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd'], help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='dann', choices=['dann', 'cosine', 'step', 'const'], 
                        help='lr scheduler of dann, cosine, step, or const')
    parser.add_argument('--decay_epoch', type=int, nargs='+', default=[80, 120], 
                        help='decrease learning rate at these epochs for step decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='lr is multiplied by gamma on decay step')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (L2 penalty)')
    parser.add_argument('--nesterov', action='store_true', help='whether to use nesterov SGD')
    parser.add_argument('--eps', type=float, default=1e-6, help='a small value to prevent underflow')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # specific optimization options
    parser.add_argument('--remain', default=0.7, type=float, help='the remaining weight of centroid of last epoch at this epoch (a number in [0, 1))')
    parser.add_argument('--temperature', default=2, type=float, help='the temperature of the softmax used for centroid classification')
    parser.add_argument('--div', type=str, default='kl', help='measure of divergence between one target instance and its perturbed counterpart')
    parser.add_argument('--gray_tar_agree', action='store_true', help='whether to enforce the consistency between RGB and gray images on the target domain')
    parser.add_argument('--aug_tar_agree', action='store_true', help='whether to enforce the consistency between RGB and augmented images on the target domain')
    parser.add_argument('--sigma', type=float, default=0.1, help='standard deviation of Gaussian')
    # checkpoints
    parser.add_argument('--start_epoch', type=int, metavar='N', default=0, help='start epoch (useful on restarts)')
    parser.add_argument('--resume', type=str, default='', help='checkpoint path to resume (default: '')')
    parser.add_argument('--eval_only', action='store_true', help='flag of evaluation only')
    # architecture
    parser.add_argument('--arch', type=str, default='resnet50', help='model name')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false', help='whether using pretrained model')
    parser.add_argument('--num_neurons', type=int, default=None, help='number of neurons in the fc1 of a new model')
    parser.add_argument('--pret_stat', type=str, default=None, help='path of pre-trained checkpoint that includes pre-trained model state dict') 
    # i/o
    parser.add_argument('--log', type=str, default='./checkpoints/', help='log folder')
    parser.add_argument('--extra_mark', type=str, default=None, help='extra mark to log folder')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--no_da', action='store_true', help='whether using data augmentation')
    parser.add_argument('--stop_epoch', default=200, type=int,
                        metavar='N', help='stop epoch (default: 200)')
    parser.add_argument('--print_freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    args = parser.parse_args()

    args.log += args.src + '2' + args.tar_tr + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + '_' + args.arch + '_pretrain' + str(args.pretrained)
    if args.pret_stat is not None:
        args.log += '_ours'
    args.log += '_' + args.extra_mark if args.extra_mark is not None else ''

    return args
