# %% 
from preprocess_data import load_and_preprocess_data

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

# Peprocess Data
coords, prices, coord_scaler, price_scaler = load_and_preprocess_data()

# Create Dataloaders
_T = lambda a: torch.tensor(a, dtype=torch.float32)
dataset = TensorDataset(_T(coords), _T(prices))

train_dl, val_dl, test_dl = [DataLoader(ds, batch_size=128) for ds in 
                             random_split(dataset, [0.7, 0.15, 0.15])]

# Print Test Output
for coords_batch, prices_batch in train_dl:
    print("Coordinates:", coords_batch.shape)
    print("Prices:", prices_batch.shape)
    break
