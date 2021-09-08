from kge.util.io import load_checkpoint
import argparse
from pprint import pprint
import os

import torch
import numpy as np
from scipy.stats import pearsonr
import wandb

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
                    default=0.0015,
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
parser.add_argument("--compute",
                    action='store_true',
                    default=False,
                    help="Whether to compute everything or try to load.")
parser.add_argument("--verbose",
                    action='store_true',
                    default=False,
                    help="Verbose mode, mainly for memory output.")
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

wandb.init(project="kginfluence", entity="mcm")
wandb.config.update(args)  # adds all of the arguments as config variables

if args.verbose:
    print(f'Memory pre model load')
    print(torch.cuda.memory_summary(device=None, abbreviated=True))
    print('###' * 30)

# load pretrained model
checkpoint = load_checkpoint('cp/toy_rescal_fb.pt')
# checkpoint = load_checkpoint('cp/toy_distmult.pt')
model = KGEInfluence.create_from(checkpoint, device).to(device)
model.damping = args.damping
model.avextol = args.avextol
model.compute = args.compute

strings_file = 'kge/data/toy/entity_strings.del'
strings_to_concepts = {}
# Open file 
with open (strings_file, "r") as fileHandler:
    # Read each line in loop
    for line in fileHandler:
        # As each line (except last one) will contain new line character, so strip that
        line = line.strip().split('\t')
        strings_to_concepts[line[0]] = line[1]

print(model)

if args.verbose:
    print(f'Memory post model load')
    print(torch.cuda.memory_summary(device=None, abbreviated=True))
    print('###' * 30)
# load data and create dataloader
train_triples = model.dataset.load_triples('train')
test_triples = model.dataset.load_triples('test')

# we need indices to get corresponding embeddings gradients
model.test_indices = test_triples

# TODO: args Parameters for dataloader
params = {'batch_size': 32, 'shuffle': False, 'num_workers': 6}
params_test = {'batch_size': 1, 'shuffle': False, 'num_workers': 6}

observed_influences = []
predicted_influences = []

# train_loader = torch.utils.data.DataLoader(train_triples, **params)
# single_train_loader = torch.utils.data.DataLoader(train_triples, **params_test)
# test_loader = torch.utils.data.DataLoader(test_triples[:10], **params_test)
# This was rewrited!
# we only want to compute train triples that contain s, r or o 
for test_triple in test_triples[:10]:
    s = test_triple[0]
    p = test_triple[1]
    o = test_triple[2]
    print(f'({s.item()}, {p.item()}, {o.item()})')

    s_all = train_triples[((train_triples[:, 0] == s) | (train_triples[:, 2] == s)).nonzero().squeeze(1)]
    r_all = train_triples[(train_triples[:, 1] == p).nonzero().squeeze(1)]
    o_all = train_triples[((train_triples[:, 0] == o) | (train_triples[:, 2] == o)).nonzero().squeeze(1)]

    sampled_train = torch.cat((s_all, r_all,o_all), dim=0)
    sampled_train = torch.unique(sampled_train, dim=0)

    train_loader = torch.utils.data.DataLoader(sampled_train, **params)
    single_train_loader = torch.utils.data.DataLoader(sampled_train, **params_test)
    test_loader = torch.utils.data.DataLoader(test_triple, **params_test)
    
    # get influences
    influences = model.get_influence(train_loader, single_train_loader,
                                    test_triple, args.verbose)

    print('Finished!')

    # get the most influential values' indices
    top_values, top_indices = torch.topk(influences, 3)
    train_indices = top_indices.clone()
    # top_indices correspond to all train data not only to sampled_train 
    
    for i, top in enumerate(top_indices):
        train_indices[i] = torch.where((train_triples == sampled_train[top]).all(dim=1))[0]

    # retrain without those samples
    print('\n\n')
    index = train_indices[0].item()
    # TODO: proper folder path
    folder_path = os.path.join(
    'retrains/rescal_fb_default/', '%s' %
    (str(index)))
    print(f'kge start kge_configs/rescal_fb_default.yaml -f {folder_path} --random_seed.default 1 --random_seed.python 2 --random_seed.torch 3 --random_seed.numpy 4 --random_seed.numba 5 --user.retrain {index}')
    if not os.path.exists(folder_path):
        os.system(f'kge start kge_configs/rescal_fb_default.yaml -f {folder_path} --random_seed.default 1 --random_seed.python 2 --random_seed.torch 3 --random_seed.numpy 4 --random_seed.numba 5 --user.retrain {index} > /dev/null 2>&1')
        print(f'Finished {index}')
        # torch.cuda.empty_cache() 
    else:
        print('Already retrained')
        
    # print('Finished retraining!')

    # compare IF and retraining (observed)
    # for i, index in enumerate(max_indices):
    print('\n\n\n')
    print(index)
    cp_path = os.path.join(
    'retrains/rescal_fb_default/', '%s/%s' %
    (str(index), 'checkpoint_best.pt'))

    checkpoint = load_checkpoint(cp_path)
    retrained_model = KGEInfluence.create_from(checkpoint, device).to(device)
    
    s = s.unsqueeze(0).to(device)
    p = p.unsqueeze(0).to(device)
    o = o.unsqueeze(0).to(device)

    # we need to unsqueeze to match scoring function :(
    print(f'({s.item()}, {p.item()}, {o.item()})')
    retrained_score = retrained_model.score_sp(s,p,o)
    score = model.score_sp(s, p, o)


    print('###' * 30)
    print(f'Full model score: {score.item()}')
    print(f'Retrained model score: {retrained_score.item()}')
    print('===' * 30)
    print('Top Standard')
    scores = model.score_sp(s, p)                # scores of all objects for (s,p,?)
    topk = torch.topk(scores, 5, dim=-1)
    print(topk)
    print('Top Retrained')
    scores = retrained_model.score_sp(s, p)                # scores of all objects for (s,p,?)
    topk = torch.topk(scores, 5, dim=-1)
    print(topk)

    print('===' * 30)
    print(f'Actual diff: {(score-retrained_score).item()}')
    print(f'Predicted inf: {influences[top_indices[0].item()]}')

    print('===' * 30)
        
    print(f'Number of training triplets containing s ({strings_to_concepts[model.dataset.entity_strings(s)[0]]}): {(train_triples.cuda()[:, 0] == s).sum() + (train_triples.cuda()[:, 2] == s).sum()}')
    print(f'Number of training triplets containing p ({model.dataset.relation_strings(p)[0]}): {(train_triples.cuda()[:, 1] == p).sum()}')
    print(f'Number of training triplets containing o ({strings_to_concepts[model.dataset.entity_strings(o)[0]]}): {(train_triples.cuda()[:, 0] == o).sum() + (train_triples.cuda()[:, 2] == o).sum()}')


    print('===' * 30)
    print(f'The most influential triples for: ({strings_to_concepts[model.dataset.entity_strings(s)[0]]}, {model.dataset.relation_strings(p)[0]}, {strings_to_concepts[model.dataset.entity_strings(o)[0]]}) where')

    for top in train_indices:
        print(top)
        print(f'({strings_to_concepts[str(model.dataset.entity_strings(train_triples[top][0]))]}, {model.dataset.relation_strings(train_triples[top][1])} , {strings_to_concepts[str(model.dataset.entity_strings(train_triples[top][2]))]})')

    # print(f'The most influential triples for: ({strings_to_concepts[model.dataset.entity_strings(s)[0]]}, {model.dataset.relation_strings(p)[0]}, {strings_to_concepts[model.dataset.entity_strings(o)[0]]}) where ({strings_to_concepts[str(model.dataset.entity_strings(train_triples[index][0]))]}, {model.dataset.relation_strings(train_triples[index][1])} , {strings_to_concepts[str(model.dataset.entity_strings(train_triples[index][2]))]})')

    print('===' * 30)

    observed_influences.append((score-retrained_score).item())
    predicted_influences.append(influences[top_indices[0].item()].item())

# print(f'Correlation is {pearsonr(predicted_influences, observed_influences)[0]}')
print(f'Observed inf: {observed_influences}')
print(f'Predicted inf: {predicted_influences}')
diffs = torch.FloatTensor(observed_influences) - torch.FloatTensor(predicted_influences)
print(diffs)
print(
    f'Correlation is {np.corrcoef(predicted_influences, observed_influences)[0, 1]}'
)

print(f'Damping: {model.damping}')
print(f'Average diff (abs) is: {torch.abs(torch.mean(diffs))} ')

wandb.log({'predicted_influences': predicted_influences})
wandb.log({'observed_influences': observed_influences})
wandb.log({'correlation': np.corrcoef(predicted_influences, observed_influences)[0, 1]})
wandb.log({'diffs': diffs})
wandb.log({'avg_diff': torch.abs(torch.mean(diffs))})
