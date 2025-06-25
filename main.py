#%%

from download_data import download_data
from preprocess_data import preprocess_data

def main():
  download_data()
  listings, reviews = preprocess_data(SLICE=(5,3), write_csv=True)

  # Print some information about the processed data
  print(f"Listings shape: {listings.shape}")
  print(f"Reviews shape: {reviews.shape}")
  print(f"You can look at the listings_clean.csv and reviews_clean.csv files in the data directory.")

if __name__ == "__main__":  # Preprocess the data
  main()  