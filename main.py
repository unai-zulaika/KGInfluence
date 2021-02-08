from kge.util.io import load_checkpoint
import argparse
import os

import torch
import numpy as np
from scipy.stats import pearsonr

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
checkpoint = load_checkpoint('cp/fb15k-237_distmult.pt')
# checkpoint = load_checkpoint('cp/toy_distmult.pt')
model = KGEInfluence.create_from(checkpoint, device).to(device)
model.train()
print(model)

# load data and create dataloader
train_triples = model.dataset.load_triples('train')
test_triples = model.dataset.load_triples('test')
# we need indices to get corresponding embeddings gradients
model.test_indices = test_triples

if args.compare:
    print('Comparing mode...')
    checkpoint = load_checkpoint('cp/toy_modified.pt')
    model_modified = KGEInfluence.create_from(checkpoint, device).to(device)
    # load data and create dataloader
    modified_train_triples = model_modified.dataset.load_triples('train')
    modified_test_triples = model_modified.dataset.load_triples('test')
    observed_influences = []

    # get all the observed diffs given a modified dataset
    for test_triple in test_triples[0:6]:
        s = test_triple[[0]].to(device)
        p = test_triple[[1]].to(device)
        o = test_triple[[2]].to(device)

        scores = model.score_sp(s, p, o)  # scores of all objects for (s,p,?)
        mod_scores = model_modified.score_sp(
            s, p, o)  # scores of all objects for (s,p,?)
        print(scores)
        print(mod_scores)
        observed_influences.append(mod_scores.item() - scores.item())
    
    exit()

# TODO: args Parameters for dataloader
params = {'batch_size': 32, 'shuffle': False, 'num_workers': 6}
params_test = {'batch_size': 1, 'shuffle': False, 'num_workers': 6}

train_loader = torch.utils.data.DataLoader(train_triples, **params)
single_train_loader = torch.utils.data.DataLoader(train_triples, **params_test)
test_loader = torch.utils.data.DataLoader(test_triples[0:6], **params_test)

# get influences
influences = model.get_influence(train_loader, single_train_loader,
                                 test_loader)

print('Finished!')

predicted_influences = []

for index, influence in enumerate(influences):
    predicted_influences.append(influence[0].item())
    print(f'Predicted diff: {influence[0].item()}')
    print(f'Actual diff: {observed_influences[index]}')
    print(
        f'Correlation is {np.corrcoef([influence[0].item()], [observed_influences[index]])[0, 1]}'
    )
    print('#' * 40)

print(f'Correlation is {pearsonr(predicted_influences, observed_influences)}')

print(
    f'Correlation is {np.corrcoef(predicted_influences, observed_influences)[0, 1]}'
)

print(f'mean actual diff {np.mean(observed_influences)}')
