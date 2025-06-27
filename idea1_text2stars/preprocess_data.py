#%%
import csv
import re
import pandas as pd
from pathlib import Path

import sys; sys.path.append(str(Path(__file__).parent.parent))
from download_files import download_and_extract, listings_url, reviews_url

def preprocess_listings(csv_path, SLICE=0, save_output=False):
  listings = pd.read_csv(csv_path, usecols=[
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
  if SLICE: listings = listings.head(SLICE)
  if save_output: listings.to_csv(Path(f'cache/listings_text2stars.csv'), index=False)  
  return listings

def preprocess_reviews(listings, csv_path, SLICE=0, save_output=False):
  reviews = pd.read_csv(csv_path, usecols=[
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

  if save_output: reviews.to_csv(Path(f'cache/reviews_text2stars.csv'), index=False)
  return reviews

def preprocess_data(SLICE=(0,0), save_output=False):
  """To see what it does, call `preprocess_data(SLICE=(5,3), save_output=True)` and look at listings_text2stars.csv and reviews_text2stars.csv"""
  listings_path = download_and_extract(listings_url)
  listings = preprocess_listings(csv_path=listings_path, SLICE=SLICE[0], save_output=save_output)

  reviews_path = download_and_extract(reviews_url)
  reviews = preprocess_reviews(listings, csv_path=reviews_path, SLICE=SLICE[1], save_output=save_output)
  
  return listings, reviews
