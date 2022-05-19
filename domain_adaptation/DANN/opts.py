import argparse


def opts():
    parser = argparse.ArgumentParser(description='DANN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source_data_path', type=str, default='/data1/Office/office_caltech_10/amazon/',
                        help='Root of source data set.')
    parser.add_argument('--target_data_path', type=str, default='/data1/Office/office_caltech_10/webcam/',
                        help='Root of target data set.')
    parser.add_argument('--test_data_path', type=str, default='/data1/Office/office_caltech_10/webcam/',
                        help='Root of test data set.')
    parser.add_argument('--source', type=str, default='A',
                        help='Source domain.')
    parser.add_argument('--target', type=str, default='W',
                        help='Target domain.')
    parser.add_argument('--num_classes', type=int, default=31,
                        help='Number of classes of data used to fine-tune the pre-trained model.')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--batch_size_s', '-b_s', type=int, default=64, help='Batch size of source data.')
    parser.add_argument('--batch_size_t', '-b_t', type=int, default=64, help='Batch size of target data.')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.01, help='Learning Rate.')
    parser.add_argument('--lr_scheduler', type=str, default='dann', choices=['dann', 'cosine', 'step', 'const'], 
                        help='lr scheduler of dann, cosine, step, or const')
    parser.add_argument('--decay_epoch', type=int, nargs='+', default=[80, 120], 
                        help='decrease learning rate at these epochs for step decay')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4, help='Weight decay (L2 penalty).')
    #parser.add_argument('--schedule', type=int, nargs='+', default=[79, 119],
    #                    help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # Checkpoints
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Manual epoch number (useful on restarts).')
    parser.add_argument('--resume', type=str, default='', help='Checkpoint path to resume (default: none).')
    parser.add_argument('--test_only', action='store_true', help='Test only flag.')
    # Architecture
    parser.add_argument('--arch', type=str, default='resnet50', help='Model name.')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false', help='Whether using pretrained model.')
    parser.add_argument('--pret_stat', type=str, default=None, help='Path of pre-trained checkpoint that includes pre-trained model state dict.')
    # I/O
    parser.add_argument('--log', type=str, default='./checkpoints/', help='Log folder.')
    parser.add_argument('--extra_mark', type=str, default=None, help='Extra mark to log folder.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers (default: 4).')
    parser.add_argument('--test_freq', default=1, type=int,
                        help='Test frequency (default: 1).')
    parser.add_argument('--stop_epoch', default=200, type=int,
                        metavar='N', help='stop epoch (default: 200)')
    parser.add_argument('--print_freq', '-p', default=10, type=int,
                        metavar='N', help='Print frequency (default: 10).')
    args = parser.parse_args()
    
    args.log += args.source + '2' + args.target + '_bs' + str(args.batch_size_s) + '_lr' + str(args.lr) + '_' + args.arch + '_pretrain' + str(args.pretrained)
    if args.pret_stat is not None:
        args.log += '_ours'
    args.log += '_' + args.extra_mark if args.extra_mark is not None else ''
        
    return args
