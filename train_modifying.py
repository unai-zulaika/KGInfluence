from kge.util.io import load_checkpoint
import argparse
import os

import torch
import numpy as np
from scipy.stats import pearsonr

from model.KGEInfluence import KGEInfluence


def dump(obj):
    for attr in dir(obj):
        if hasattr(obj, attr):
            print("obj.%s = %s" % (attr, getattr(obj, attr)))


parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    type=str,
                    default="FB15k-237",
                    nargs="?",
                    help="Which dataset to use: FB15k-237 or WN18RR.")
parser.add_argument("--num_iterations",
                    type=int,
                    default=500,
                    nargs="?",
                    help="Number of iterations.")
parser.add_argument("--batch_size",
                    type=int,
                    default=128,
                    nargs="?",
                    help="Batch size.")
parser.add_argument("--nneg",
                    type=int,
                    default=50,
                    nargs="?",
                    help="Number of negative samples.")
parser.add_argument('--avextol',
                    type=float,
                    default=1e-03,
                    help='threshold for optimization in influence function')
parser.add_argument('--damping',
                    type=float,
                    default=1e-01,
                    help='damping term in influence function')
parser.add_argument("--lr",
                    type=float,
                    default=50,
                    nargs="?",
                    help="Learning rate.")
parser.add_argument("--dim",
                    type=int,
                    default=40,
                    nargs="?",
                    help="Embedding dimensionality.")
parser.add_argument("--no-wandb",
                    action='store_true',
                    default=False,
                    help="Log wandb.")
parser.add_argument("--compare", action='store_true', help=".")
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')

args = parser.parse_args()

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# fix seeds
torch.backends.cudnn.deterministic = True
seed = 40
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available:
    torch.cuda.manual_seed_all(seed)

# track experiments with wandb
if args.no_wandb:
    os.environ['WANDB_MODE'] = 'dryrun'

# load pretrained model
checkpoint = load_checkpoint('cp/toy.pt')
# checkpoint = load_checkpoint('cp/toy_distmult.pt')
model = KGEInfluence.create_from(checkpoint, device).to(device)
model.train()
print(model)

# load data and create dataloader
train_triples = model.dataset.load_triples('train')
test_triples = model.dataset.load_triples('test')
# we need indices to get corresponding embeddings gradients
model.test_indices = test_triples

# TODO: args Parameters for dataloader
params = {'batch_size': 32, 'shuffle': False, 'num_workers': 6}
params_test = {'batch_size': 1, 'shuffle': False, 'num_workers': 6}

train_loader = torch.utils.data.DataLoader(train_triples, **params)
single_train_loader = torch.utils.data.DataLoader(train_triples, **params_test)
test_loader = torch.utils.data.DataLoader(test_triples[0:6], **params_test)

# print(model.dataset)

# attrs = vars(model.dataset)
# print(', '.join("%s: %s" % item for item in attrs.items()))

for i in range(model.dataset._triples['train'].shape[0]):
    if i == 0:
        print(model.dataset._triples['train'][1:, :].shape)
    elif i+1 == model.dataset._triples['train'].shape[0]:
        print(model.dataset._triples['train'][:i, :].shape)
        print("LAST")
    else:
        print(
            torch.cat([
                model.dataset._triples['train'][0:i],
                model.dataset._triples['train'][i + 1:]
            ]).shape)

print(model.dataset._triples['train'].shape)

# kge start kge_configs/train_config.yaml -f retrains/test2 --random_seed.default 1 --random_seed.python 2 --random_seed.torch 3 --random_seed.numpy 4 --random_seed.numba 5