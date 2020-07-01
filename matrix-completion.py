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
    test_df['rating'].replace({"?": 0}, inplace=True)
    return train_df, test_df

def shuffle_data(df):
    df = shuffle(df)
    return df.reset_index(drop=True)

def make_matrix(train_df, train_cid, train_mid):
    train_movies_n = len(train_mid)
    train_cust_n = len(train_cid)
    # Make movie-by-user matrix to hold ratings
    ratings = np.zeros((train_cust_n, train_movies_n))

    for i in range(len(train_df)):
        current = train_df.iloc[i]
        customer_idx = train_cid.get(current[1])
        movie_idx = train_mid.get(current[0])
        ratings[customer_idx, movie_idx] = current[2]
    
    return ratings

def construct_dict(df, column):
    col = df[column]
    col_sorted = col.sort_values()
    d = {}
    cursor = 0
    for val in col_sorted:
        if val not in d:
            d[val] = cursor
            cursor += 1
    return d

# main
validate = True
base_path = os.getcwd()
train_df, test_df = load_data(base_path)
train_df = shuffle_data(train_df)

if validate:
    split_index = round(0.1 * train_df.shape[0])
    validate_df = train_df[: split_index]
    val_df_original = validate_df.copy()
    validate_df['rating'].values[:] = 0


# construct a dict to map users, movies with row, column ids
train_cid = construct_dict(train_df, 'customer-id')
train_mid = construct_dict(train_df, 'movie-id')

ratings = make_matrix(train_df, train_cid, train_mid)