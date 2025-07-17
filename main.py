# %% Imports
from preprocess_data import load_and_preprocess_data
from siren import *

import torch 
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt

torch.manual_seed(42) # For replicability

DRAFTING = not torch.cuda.is_available() # if we only have CPU only check if it runs at all. Do proper training on HPC GPU

# %% Data Preprocessing and Loading
coords, prices, coord_scaler, price_scaler = load_and_preprocess_data()

_T = lambda a: torch.tensor(a, dtype=torch.float32)
dataset = TensorDataset(_T(coords), _T(prices))

train_dl, val_dl, test_dl = [DataLoader(ds, batch_size=128) for ds in 
                             random_split(dataset, [0.7, 0.15, 0.15])]

# %% Plot how many annotations we have where
sns.kdeplot(x=coords[:,0], y=coords[:,1], fill=True)
plt.title("Density of listings in coordinate space")
plt.show()

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

# Plot Learning Rate and Loss
sns.lineplot(x=range(len(lrs)), y=lrs, label='Learning Rate')
plt.title("Learning Rate over Iterations")
plt.show()

sns.lineplot(x=range(len(losses)), y=losses, label='Loss')
plt.title("Training Loss over Iterations")
plt.show()

# %% Plot Predicted Price Map on Hong Kong Basemap
coords, prices, coord_scaler, price_scaler = load_and_preprocess_data()
coords_tensor = torch.tensor(coords, dtype=torch.float32)
prices_tensor = torch.tensor(prices, dtype=torch.float32)
model.eval()
with torch.no_grad():
    preds = model(coords_tensor).squeeze().numpy()
    preds = np.clip(preds, 0, 1)  # clip predictions if target scaled [0,1]

df = pd.DataFrame({
    'latitude': coords[:,0],
    'longitude': coords[:,1],
    'predicted_price': preds,
})

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
).to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(12, 10))
gdf.plot(
    ax=ax,
    column='predicted_price',
    cmap='cividis',
    markersize=30,
    alpha=0.7,
    legend=True,
    legend_kwds={'label': 'Predicted Price (scaled)'}
)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
ax.set_axis_off()
plt.title("Airbnb Predicted Price Map Over Hong Kong")
plt.show()