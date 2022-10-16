import os
import argparse

USE_CUDA = True

parser = argparse.ArgumentParser(description='Parameters for DialogSum dataset')

parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5)
parser.add_argument('-cd', '--cuda', type=int, default=0)
parser.add_argument('-sd', '--seed', type=int, default=1919810)
parser.add_argument('-eps', '--epochs', type=int, default=15)
parser.add_argument('-mgr', '--max_grad_norm', type=float, default=1.0)
parser.add_argument('-dp', '--data_path', type=str, default='data')
parser.add_argument('-mt', '--model_type', type=str, default='bart')
parser.add_argument('-cp', '--cache_path', type=str, default='cache')
parser.add_argument('-ml', '--max_length', type=int, default=512)
parser.add_argument('-sml', '--summary_max_length', type=int, default=100)
parser.add_argument('-bsz', '--batch_size', type=int, default=12)
parser.add_argument('-dbg', '--debug', type=bool, default=False)
parser.add_argument('-wmprop', '--warmup_proportion', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--small', type=bool, default=False, help='whether to use small dataset')
parser.add_argument('--save_path', type=str, default='save')
parser.add_argument('--dataset', type=str, default='dialogsum', choices=['dialogsum', 'samsum'])

# parser.add_argument('--model_name', type=str, default='lidiya/bart-large-xsum-samsum')
parser.add_argument('--model_name', type=str, default='facebook/bart-large')
parser.add_argument('--num_beams', type=int, default=4)
parser.add_argument('--model_file', type=str, default='baseline')
parser.add_argument('--colab', type=int, default=0)
parser.add_argument('--num_decouple_layers', type=int, default=1)
parser.add_argument('--avg', type=int, default=0, help='whether to use average scores over all references')

parser.add_argument('--fp16', type=int, default=1)

args = parser.parse_args()

save_root = '{}_saves'.format(args.dataset) if not args.small else '{}_saves_small'.format(args.dataset)
cache_root = '{}_caches'.format(args.dataset) if not args.small else '{}_caches_samll'.format(args.dataset)

if not os.path.exists(save_root):
    os.mkdir(save_root)
if not os.path.exists(cache_root):
    os.mkdir(cache_root)

if args.model_file == 'baseline': args.num_decouple_layers = 0

args.save_path = save_root + '/' + args.model_type +\
     '_' + '{}L{}lr{}'.format(args.model_file, args.num_decouple_layers,\
     args.learning_rate) + '_' + args.save_path
args.cache_path = cache_root + '/' + args.model_type + '_' + args.cache_path

args.cache_path += '_' + str(args.max_length)
args.save_path += '_' + str(args.max_length) + '_' + str(args.seed)
