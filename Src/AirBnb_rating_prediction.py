import pandas as pd
import os
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

plt.style.use('fivethirtyeight')
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')

fileDir = os.path.dirname(__file__)
listingsDir = os.path.join(fileDir, '../Data/listings.csv')
reviewsDir = os.path.join(fileDir, '../Data/reviews.csv')
listings = pd.read_csv(listingsDir)
reviews = pd.read_csv(reviewsDir)
listings = pd.DataFrame(listings)
number = 250
print(listings['review_scores_rating'].describe())
listings['review_scores_rating'].hist(bins=30)

listing_reviews = (listings
                  .set_index("id")
                  .join(reviews.set_index("listing_id"),
                        how="left")
                 )
print(listing_reviews.info)
df_x=reviews["comments"]                 
print(listing_reviews.head(2))               
#for i in range(0,number):
#print(listings.id.astype(int))

#listingID = listings[i].id
#if listingID == 
