from tokenization import ShakespeareData
from torch.utils.data import DataLoader

dataset = ShakespeareData()
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

x = iter(dataloader)
print(next(x))