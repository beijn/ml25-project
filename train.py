#%% 
# TODO Beni yayyy
# 
from preprocess_data import preprocess_data
from siren import Siren

import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
from composer import Trainer
from composer.algorithms import LabelSmoothing, CutMix, ChannelsLast
from composer.loggers import InMemoryLogger
from composer.models import ComposerModel
import composer.optim
import seaborn as sns
import matplotlib.pyplot as plt

torch.manual_seed(42) # For replicability

DRAFTING = not torch.cuda.is_available() # if we only have CPU only check if it runs at all. Do proper training on HPC GPU

class SIREN(ComposerModel):
  def __init__(self):
    super().__init__()
    self.model = Siren([2, *([100,100] if DRAFTING else [256, 128, 64, 32]), 1])
  def forward(self, batch): return self.model(batch[0])
  def loss(self, outputs, batch):
    _, targets = batch
    return nn.MSELoss()(outputs, targets)
class DummyDataset(Dataset):
  def __init__(self, size=1000):
    self.size = size
    self.data = torch.rand(size, 2) * 10  # Random coordinates in a 10x10 grid
    self.targets = torch.sin(self.data[:, 0]) + torch.cos(self.data[:, 1])  # Dummy target values
  def __len__(self):
    return self.size
  def __getitem__(self, idx):
    return self.data[idx], self.targets[idx].unsqueeze(0)  # Return data and target as a tuple

dataset = DummyDataset()
train_dataloader = DataLoader(dataset, batch_size=128)
test_dataloader = DataLoader(dataset, batch_size=128)

import numpy as np
out = np.zeros((10,10))

for xy, v in train_dataloader:
  out[xy[:,0].long(), xy[:,1].long()] = v.numpy().flatten()
  
sns.heatmap(out, cmap='viridis', cbar=True, square=True, xticklabels=False, yticklabels=False)  

# %%


model = SIREN()
#%%
optimizer = composer.optim.DecoupledSGDW(
    model.parameters(), # Model parameters to update
    lr=50000000, # Peak learning rate
    momentum=0.9,
    weight_decay=2.0e-3 # If this looks large, it's because its not scaled by the LR as in non-decoupled weight decay
)

lr_scheduler = composer.optim.LinearWithWarmupScheduler(
    t_warmup="10ep", # Warm up over 1 epoch
    alpha_i=0.5, # Flat LR schedule achieved by having alpha_i == alpha_f
    alpha_f=0.5
)

logger_for_baseline = InMemoryLogger()

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    eval_interval="1ep", # Evaluate every epoch
    max_duration="100ep" if DRAFTING else "300ep",
    optimizers=optimizer,
    #schedulers=lr_scheduler,
    device = "gpu" if torch.cuda.is_available() else "cpu", # select the device
    loggers=logger_for_baseline,
    algorithms=[],
)


trainer.fit()

x = torch.linspace(0, 10, 10)
y = torch.linspace(0, 10, 10)

pred = np.zeros((10, 10))

with torch.no_grad():
  for x in range(10):
    for y in range(10):
      pred[x, y] = model.model(torch.tensor([[x, y]], dtype=torch.float32)).item()

sns.heatmap(pred, cmap='viridis', cbar=True, square=True, xticklabels=False, yticklabels=False)
plt.show() 

timeseries_raw = logger_for_baseline.get_timeseries("loss/train/total")
plt.plot(timeseries_raw['epoch'], timeseries_raw["loss/train/total"])
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Accuracy per epoch with Baseline Training")
plt.show()
