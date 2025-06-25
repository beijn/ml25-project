#%%
import re
import pandas as pd

SLICE = 5

def preprocess_listings(path= 'data/listings.csv', SLICE=0, write_csv=False):
  listings = pd.read_csv(path, usecols=[
    'id',
    'review_scores_rating', 
    #'name',  # these others are maybe also interesting later
    #'number_of_reviews',
    #'review_scores_accuracy',
    #'review_scores_cleanliness',
    #'review_scores_checkin',
    #'review_scores_communication',
    #'review_scores_location',
    #'review_scores_value',
    #'reviews_per_month'
  ])
  listings.rename(columns={'id': 'listing_id'}, inplace=True)
  _len0 = len(listings)
  listings.dropna(inplace=True)
  _len1 = len(listings)
  print(f"Kept {(_len1)} listings without NA values out of {_len0} in total.")
  # slice to 5 rows if SLICE:
  if SLICE: listings = listings.head(SLICE)
  if write_csv: listings.to_csv('data/listings_clean.csv', index=False)  
  return listings

def preprocess_reviews(listings, path='data/reviews.csv', SLICE=0, write_csv=False):
  reviews = pd.read_csv(path, usecols=[
    'listing_id',
    'id',
    'comments',
  ])
  reviews.rename(columns={'id': 'review_id'}, inplace=True)
  _len0 = len(reviews)
  reviews.dropna(inplace=True)
  _len1 = len(reviews)
  print(f"Kept {_len1} reviews without NA values out of {_len1} in total.")

  # filter out reviews that are not associated with a listing
  _len0 = len(reviews)
  reviews = reviews[reviews['listing_id'].isin(listings['listing_id'])]
  _len1 = len(reviews)
  print(f"Kept {_len1} reviews that are associated with a listing out of {_len0} in total.")

  # normalize html in comments
  reviews['comments'] = reviews['comments'].apply(lambda x: re.sub(r'<[^>]*>', '', x))  # remove html tags
  reviews['comments'] = reviews['comments'].apply(lambda x: re.sub(r'\s+', ' ', x))  # remove extra whitespace
  reviews['comments'] = reviews['comments'].apply(lambda x: x.strip())  # strip

  # remove non-english reviews 
  _len0 = len(reviews)
  reviews = reviews[reviews['comments'].str.contains(r'^[\x00-\x7F]+$', na=False)]  # keep only ascii characters
  _len1 = len(reviews)
  print(f"Kept {_len1} reviews that are ascii only out of {_len0} in total. (As a simple heuristic of filtering for only english reviews.)")

  # keep only SLICE reviews per listing
  if SLICE:
    reviews = reviews.groupby('listing_id').head(SLICE)
    print(f"Kept {len(reviews)} reviews, with max {SLICE} per listing.")

  if write_csv: reviews.to_csv('data/reviews_clean.csv', index=False)
  return reviews

def preprocess_data(SLICE=(0,0), write_csv=False):
  listings = preprocess_listings(SLICE=SLICE[0], write_csv=write_csv)
  reviews = preprocess_reviews(listings, SLICE=SLICE[1], write_csv=write_csv)
  return listings, reviews
