import os
import argparse

USE_CUDA = True

parser = argparse.ArgumentParser(description='Parameters for MuTual dataset')

parser.add_argument('-lr', '--learning_rate', type=float, default=6e-6)
parser.add_argument('-ml', '--max_length', type=int, default=320)
parser.add_argument('-oml', '--option_max_length', type=int, default=52)
parser.add_argument('-cd', '--cuda', type=int, default=0)
parser.add_argument('-sd', '--seed', type=int, default=1919810)
parser.add_argument('-eps', '--epochs', type=int, default=3)
parser.add_argument('-mgr', '--max_grad_norm', type=float, default=1.0)
parser.add_argument('-dp', '--data_path', type=str, default='data')
parser.add_argument('-mt', '--model_type', type=str, default='electra')
parser.add_argument('-mn', '--model_name', type=str, default='google/electra-large-discriminator')
parser.add_argument('-cp', '--cache_path', type=str, default='cache')
parser.add_argument('-mun', '--max_utterance_num', type=int, default=21)
parser.add_argument('-bsz', '--batch_size', type=int, default=2)
parser.add_argument('-wmprop', '--warmup_proportion', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--adam_epsilon', type=float, default=1e-6)
parser.add_argument('--small', type=bool, default=False, help='whether to use small dataset')
parser.add_argument('--save_path', type=str, default='save')
parser.add_argument('--colab', type=bool, default=False)

parser.add_argument('--model_file', type=str, default='baseline')
parser.add_argument('--num_decouple_layers', type=int, default=1)
parser.add_argument('--three_channels', type=int, default=1)
parser.add_argument('--dataset', type=str, default='mutual')

args = parser.parse_args()


save_root = args.dataset + '_saves'
if args.small: save_root += '_small'
if not os.path.exists(save_root): os.mkdir(save_root)
if not os.path.exists('caches') and not args.small: os.mkdir('caches')
if not os.path.exists('caches_small') and args.small: os.mkdir('caches_small')
if args.model_file in ['baseline', 'baseline+BiGRU']:
    args.num_decouple_layers = 0

args.save_path = save_root + '/' + args.model_type +\
     '_' + '{}L{}lr{}'.format(args.model_file, args.num_decouple_layers, args.learning_rate) + '_' + args.save_path
args.cache_path = ('caches/' if not args.small else 'caches_small/') + args.dataset + '_' + args.model_type + '_' + args.cache_path

assert args.model_type in args.model_name
args.cache_path += '_' + str(args.max_length)
args.save_path += '_' + str(args.max_length)
