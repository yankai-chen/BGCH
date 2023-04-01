import os
import sys
from os.path import join
import torch
from src.parse import parse_args
from pprint import pprint

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()
CODE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = join(CODE_PATH, '../dataset')
C_SOURCE_PATH = join(CODE_PATH, 'sources')
BOARD_PATH = join(CODE_PATH, 'results')
FILE_PATH = join(CODE_PATH, 'checkpoints')
sys.path.append(C_SOURCE_PATH)

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


all_dataset = ['music', 'gowalla', 'yelp', 'book', 'movie', 'pinterest', 'dianping']
all_model = ['bgch']
GPU = torch.cuda.is_available()
SEED = args.seed
DEVICE = torch.device("cuda" if GPU else 'cpu')

FS_N = args.N
FS_w = args.w


if args.dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {args.dataset} yet!, try {all_dataset}")

from warnings import simplefilter
import warnings

warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")


print('===========config================')
pprint(args)
print('===========end===================')

