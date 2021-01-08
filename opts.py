import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str,
                        help='Directory path of Videos')
    parser.add_argument('--dataset_file', default='',
                        type=str, help='dataset file path')
    parser.add_argument('--result_path', default='results',
                        type=str, help='Result directory path')
    parser.add_argument('--dataset', default='ins',
                        type=str, help='Used dataset (ins | ucf101 | hmdb51)')
    parser.add_argument('--test_subset', default='val',
                        type=str, help='validation set or test set for testing')
    parser.add_argument('--n_classes', default=256,
                        type=int, help='Number of classes (ucf101: 101, hmdb51: 51)')
    parser.add_argument('--n_finetune_classes', default=101,
                        type=int, help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    parser.add_argument('--sample_size', default=224,
                        type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16,
                        type=int, help='Temporal duration of inputs')
    parser.add_argument('--stride_size', default=1, type=int,
                        help='Temporal stride of inputs')

    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')

    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--lr_bert', default=3e-4, type=float,
                        help='Initial learning rate for BERT')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=1e-3,
                        type=float, help='Weight Decay')
    parser.add_argument('--lr_patience', default=10, type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--lr_factor', default=0.1, type=float)
    parser.add_argument('--clip_gradient', default=20, type=float,
                        help='Clip the gradient when gradient is larger than threshold')
    parser.add_argument('--batch_size', default=128,
                        type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=200, type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--ft_bert_ep', default=150, type=int,
                        help='Epoch start to finetune bert model.')

    parser.add_argument('--resume_path', default='', type=str,
                        help='Save data (.pth) of previous training')
    parser.add_argument('--pretrain_path', default=None,
                        type=str, help='Pretrained model (.pth)')
    parser.add_argument('--ft_begin_index', default=0, type=int,
                        help='Begin block index of fine-tuning')
    parser.add_argument('--no_train', default=False, action='store_true',
                        help='If true, training is not performed.')
    parser.add_argument('--no_val', default=False, action='store_true',
                        help='If true, validation is not performed.')
    parser.add_argument('--test', default=False, action='store_true',
                        help='If true, test is performed.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--n_threads', default=4, type=int,
                        help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint', default=2, type=int,
                        help='Trained model is saved at every this epochs.')

    parser.add_argument('--model', default='resnet', type=str,
                        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth', default=50, type=int,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='A', type=str,
                        help='Shortcut type of resnet (A | B)')
    parser.add_argument('--dp', default=0., type=float,
                        help='possibility to set to zero.')
    parser.add_argument('--manual_seed', default=1, type=int,
                        help='Manually set random seed')
    parser.add_argument('--phase', default="finetuning", type=str,
                        help='pretraining or finetuning phase')
    parser.add_argument('--emb_dim', default=256, type=int,
                        help='feature dimmension in embedding space')
    parser.add_argument('--print_frq', default=100, type=int,
                        help='frequency of print training log')
    parser.add_argument('--nce_k', default=16392, type=int,
                        help='number of negative samples')
    parser.add_argument('--nce_t', default=0.07, type=float,
                        help='temperature that modulates the possible distribution')
    parser.add_argument('--nce_m', default=0.5, type=float,
                        help='the momentum for dynamically updating the memory')

    args = parser.parse_args()

    return args
