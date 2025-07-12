import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from download_files import download_and_extract, listings_url

def load_and_preprocess_data():
  csv_path = download_and_extract(listings_url)
  df = pd.read_csv(csv_path)[['latitude', 'longitude', 'price']].dropna()
    
  coords = df[['latitude', 'longitude']].values.astype(np.float32)
  prices = df['price'].apply(lambda s: s.removeprefix('$').replace(',','')).values.astype(np.float32).reshape((-1,1))/100

  coord_scaler = MinMaxScaler((0,1))
  price_scaler = MinMaxScaler((0,1))
  
  coords_scaled = coord_scaler.fit_transform(coords)
  prices_scaled = price_scaler.fit_transform(prices)

  return coords_scaled, prices_scaled, coord_scaler, price_scaler
