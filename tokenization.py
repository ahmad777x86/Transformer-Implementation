from torch.utils.data import Dataset
import torch

class ShakespeareData(Dataset):
    def __init__(self, context_window_size = 3, path = 'corpus.txt'):
        super().__init__()
        data = open(path).read()
        self.corpus = list(data)
        self.length = len(list(data))
        self.vocab = sorted(list(set(data)))
        self.block_size = context_window_size

        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        
    def __len__(self):
        return self.length - self.block_size - 2

    def __getitem__(self, index):
        input_seq = torch.tensor([self.stoi[ch] for ch in self.corpus[index : index+self.block_size]])
        target_word = torch.tensor([self.stoi[self.corpus[index+self.block_size]]])
        return input_seq, target_word