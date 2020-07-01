import pandas as pd 
import matplotlib.pyplot as plt 
import io, csv, os
import numpy as np
from collections import defaultdict
import pickle
import sklearn.metrics
from sklearn.utils import shuffle
from scipy.sparse.linalg import svds
import math

def load_data(base_path):
    train_df = pd.read_csv( base_path + "train.csv")
    test_df = pd.read_csv( base_path + "test.csv")
    return 

raw_df = shuffle(raw_df)
raw_df = raw_df.reset_index(drop=True)

split_index = round(0.1 * raw_df.shape[0])
validate_df = raw_df[: split_index]

val_df_original = validate_df.copy()
validate_df['rating'].values[:] = 0

test_df['rating'].replace({"?": 0}, inplace=True)

combined_df = raw_df

# construct a dict to map users, movies with row, column ids
combined_customers = combined_df['customer-id']
combined_cid_sorted = combined_customers.sort_values()
combined_movies = combined_df['movie-id']
combined_mid_sorted = combined_movies.sort_values()

combined_cid = {}
cursor = 0
for val in combined_cid_sorted:
    if val not in combined_cid:
        combined_cid[val] = cursor
        cursor += 1

combined_mid = {}
cursor = 0
for val in combined_mid_sorted:
    if val not in combined_mid:
        combined_mid[val] = cursor
        cursor += 1

combined_movies_n = len(combined_mid)
combined_cust_n = len(combined_cid)
print(combined_movies_n, combined_cust_n)

# Make movie-by-user matrix to hold ratings
ratings = np.zeros((combined_cust_n, combined_movies_n))

for i in range(len(combined_df)):
    current = combined_df.iloc[i]
    customer_idx = combined_cid.get(current[1])
    movie_idx = combined_mid.get(current[0])
    ratings[customer_idx, movie_idx] = current[2]