from kge.util.io import load_checkpoint
import argparse

import torch
import numpy as np

from model.KGEInfluence import KGEInfluence

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
                    default=1e-3,
                    help='threshold for optimization in influence function')
parser.add_argument('--damping',
                    type=float,
                    default=1e-6,
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
parser.add_argument("--cuda",
                    type=bool,
                    default=True,
                    nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")

args = parser.parse_args()

# TODO: device from args
device = torch.device('cuda')  # GPU

dataset = args.dataset
data_dir = "data/%s/" % dataset

# fix seeds
torch.backends.cudnn.deterministic = True
seed = 40
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available:
    torch.cuda.manual_seed_all(seed)

# load pretrained model
checkpoint = load_checkpoint('cp/fb15k-237-rescal.pt')
model = KGEInfluence.create_from(checkpoint, device).to(device)

# s = torch.Tensor([
#     0,
#     2,
# ]).long()  # subject indexes
# p = torch.Tensor([
#     0,
#     1,
# ]).long()  # relation indexes
# scores = model.score_sp(s, p)  # scores of all objects for (s,p,?)
# max_score = torch.max(scores, dim=-1)  # get max score value
# o = torch.argmax(scores, dim=-1)  # index of highest-scoring objects

# print('==' * 30)

# print(max_score)
# print(model.dataset.entity_strings(s))  # convert indexes to mentions
# print(model.dataset.relation_strings(p))
# print(model.dataset.entity_strings(o))

# print('***' * 30 + '\n')

# load data and create dataloader
train_triples = model.dataset.load_triples('train')
test_triples = model.dataset.load_triples('test')

# TODO: args Parameters for dataloader
params = {'batch_size': 64, 'shuffle': False, 'num_workers': 6}
params_test = {'batch_size': 1, 'shuffle': False, 'num_workers': 6}

train_loader = torch.utils.data.DataLoader(train_triples, **params)
test_loader = torch.utils.data.DataLoader(test_triples[0:4, :], **params_test)

# get influence
model.get_influence(train_loader, test_loader)
