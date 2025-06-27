# NOTE this currently only filters the data
#%%

from preprocess_data import preprocess_data

def main():
  listings, reviews = preprocess_data(SLICE=(5,3), save_output=True)

  # Print some information about the processed data
  print(f"Listings shape: {listings.shape}")
  print(f"Reviews shape: {reviews.shape}")
  print(f"You can look at the listings_text2stars.csv and reviews_text2stars.csv files in the data directory.")

if __name__ == "__main__":  # Preprocess the data
  main()  
  