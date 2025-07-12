# %% Imports
from preprocess_data import load_and_preprocess_data
from siren import *

import torch 
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np, seaborn as sns, matplotlib.pyplot as plt, time

torch.manual_seed(42) # For replicability

DRAFTING = not torch.cuda.is_available() # if we only have CPU only check if it runs at all. Do proper training on HPC GPU

# %% Data Preprocessing and Loading
coords, prices, coord_scaler, price_scaler = load_and_preprocess_data()

_T = lambda a: torch.tensor(a, dtype=torch.float32)
dataset = TensorDataset(_T(coords), _T(prices))

train_dl, val_dl, test_dl = [DataLoader(ds, batch_size=128) for ds in 
                             random_split(dataset, [0.7, 0.15, 0.15])]

# %% Plot how many annotations we have where
sns.kdeplot( x=coords[:,0], y=coords[:,1],
    fill=True,
)

# %% Model Definition
model = Siren([2, 128, 64, 32, 1])

# %% Training
epochs = 10 if DRAFTING else 300

lossf = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(), # Model parameters to update
    lr=1e-3, # Peak learning rate
)

lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, 
    start_factor=1, 
    end_factor=0.1,
    total_iters=epochs,
)

start_time = time.perf_counter()
lrs, losses = [], []
for e in range(epochs):
  for X,Y in train_dl:
    optimizer.zero_grad()
    Z = model(X)
    loss = lossf(Z, Y)
    loss.backward()
    optimizer.step()
    lrs.append(optimizer.param_groups[0]['lr'])
    losses.append(loss.item())
  if lr_scheduler is not None:
    lr_scheduler.step()
  
  print(f"Epoch {e+1}, Loss: {loss.item():.4f}")

end_time = time.perf_counter()
print(f"It took {end_time - start_time:0.4f} seconds to train")

plt.show()
sns.lineplot(x=range(len(lrs)), y=lrs, label='Learning Rate')
plt.show()
sns.lineplot(x=range(len(losses)), y=losses, label='Loss')
plt.show()

# %% 
# TODO Alina plot price map on top of Hong Kong map
