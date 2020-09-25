import torch

class KGDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, triples):
        'Initialization'
        self.scoring = triples[:, :2]
        self.labels = triples[:, 2:]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.scoring)

  def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.scoring[index]
        y = self.labels[index]

        return X, y