import argparse

def opts():
    parser = argparse.ArgumentParser(description='SRDC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # datasets
    parser.add_argument('--data_path_source', type=str, default='./data/datasets/Office31/', help='root of source training set')
    parser.add_argument('--data_path_target', type=str, default='./data/datasets/Office31/', help='root of target training set')
    parser.add_argument('--data_path_target_t', type=str, default='./data/datasets/Office31/', help='root of target test set')
    parser.add_argument('--src', type=str, default='amazon', help='source training set')
    parser.add_argument('--tar', type=str, default='webcam_half', help='target training set')
    parser.add_argument('--tar_t', type=str, default='webcam_half2', help='target test set')
    parser.add_argument('--num_classes', type=int, default=31, help='class number')
    # source sample selection
    parser.add_argument('--src_soft_select', action='store_true', help='whether to softly select source instances')
    parser.add_argument('--src_mix_weight', action='store_true', help='whether to mix 1 and soft weight')
    parser.add_argument('--src_hard_select', action='store_true', help='whether to hardly select source instances')
    parser.add_argument('--tao', type=float, default=0.5, help='threshold parameter of cosine similarity')
    # general optimization options
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')    
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--workers', type=int, default=8, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--no_da', action='store_true', help='whether to not use data augmentation')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd'], help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='dann', choices=['dann', 'cosine', 'step', 'const'], 
                        help='lr scheduler of dann, cosine, step, or const')
    parser.add_argument('--decay_epoch', type=int, nargs='+', default=[80, 120], 
                        help='decrease learning rate at these epochs for step decay')
    parser.add_argument('--mu', type=float, default=0.1, help='lr is multiplied by mu on decay step')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (L2 penalty)')
    parser.add_argument('--nesterov', action='store_true', help='whether to use nesterov SGD')
    parser.add_argument('--eps', type=float, default=1e-6, help='a small value to prevent underflow')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # specific optimization options
    parser.add_argument('--ao', action='store_true', help='whether to use alternative optimization')
    parser.add_argument('--cluster_method', type=str, default='kmeans', help='clustering method of kmeans or spherical_kmeans or kernel_kmeans to choose')
    parser.add_argument('--cluster_iter', type=int, default=5, help='number of iterations of K-means')
    parser.add_argument('--cluster_kernel', type=str, default='rbf', help='kernel to choose when using kernel K-means')
    parser.add_argument('--gamma', type=float, default=None, help='bandwidth for rbf or polynomial kernel when using kernel K-means')
    parser.add_argument('--sample_weight', action='store_true', help='whether to adapt sample weight when using kernel K-means')
    parser.add_argument('--initial_cluster', type=int, default=1, choices=[0, 1, 2], help='target (1) or source (2) class centroids for initialization of K-means, 0 for no k-means clustering')
    parser.add_argument('--init_cen_on_st', action='store_true', help='whether to initialize learnable cluster centers on both source and target instances')
    parser.add_argument('--src_cen_first', action='store_true', help='whether to use source class centroids as initial target cluster centers at the first epoch')
    parser.add_argument('--src_cls', action='store_true', help='whether to classify source instances when clustering target instances')
    parser.add_argument('--src_fit', action='store_true', help='whether to use convex combination of true label vector and predicted label vector as training guide')
    parser.add_argument('--src_pretr_first', action='store_true', help='whether to perform clustering over features extracted by source pre-trained model at the first epoch')
    parser.add_argument('--learn_embed', action='store_true', help='whether to apply embedding clustering')
    parser.add_argument('--no_second_embed', action='store_true', help='whether to not apply embedding clustering on output features of the first FC layer')
    parser.add_argument('--alpha', type=float, default=1.0, help='degrees of freedom of Student\'s t-distribution')
    parser.add_argument('--beta', type=float, default=1.0, help='weight of auxiliary target distribution or assigned cluster labels')
    parser.add_argument('--embed_softmax', action='store_true', help='whether to use softmax to normalize soft cluster assignments for embedding clustering')
    parser.add_argument('--div', type=str, default='kl', help='measure of prediction divergence between one target instance and its perturbed counterpart')
    parser.add_argument('--gray_tar_agree', action='store_true', help='whether to enforce the consistency between RGB and gray images on the target domain')
    parser.add_argument('--aug_tar_agree', action='store_true', help='whether to enforce the consistency between RGB and augmented images on the target domain')
    parser.add_argument('--sigma', type=float, default=0.1, help='standard deviation of Gaussian for data augmentation operation of blurring')
    # checkpoints
    parser.add_argument('--resume', type=str, default='', help='checkpoints path to resume')
    parser.add_argument('--log', type=str, default='./checkpoints/', help='log folder')
    parser.add_argument('--extra_mark', type=str, default=None, help='extra mark to log folder')
    parser.add_argument('--stop_epoch', type=int, default=200, metavar='N', help='stop epoch for early stop (default: 200)')
    # architecture
    parser.add_argument('--arch', type=str, default='resnet50', help='model name')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false', help='whether using pretrained model')
    parser.add_argument('--num_neurons', type=int, default=None, help='number of neurons in the fc1 of a new model')
    parser.add_argument('--pret_stat', type=str, default=None, help='path of pre-trained checkpoint that includes pre-trained model state dict')
    # i/o
    parser.add_argument('--print_freq', type=int, default=10, metavar='N', help='print frequency (default: 10)')

    args = parser.parse_args()
    args.src_cls = True
    args.src_cen_first = True
    args.learn_embed = True
    args.embed_softmax = True
    
    args.log += args.src + '2' + args.tar + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + '_' + args.arch + '_pretrain' + str(args.pretrained)
    if args.pret_stat is not None:
        args.log += '_ours'
    args.log += '_' + args.extra_mark if args.extra_mark is not None else ''

    return args
