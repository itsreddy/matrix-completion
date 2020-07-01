import pandas as pd 
import matplotlib.pyplot as plt 
import io, csv, os
import numpy as np
from collections import defaultdict
import sklearn.metrics
from sklearn.utils import shuffle
from scipy.sparse.linalg import svds
from scipy.linalg import svd
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
    # Make movie-by-user matrix to hold ratings
    train_movies_n = len(train_mid)
    train_cust_n = len(train_cid)
    ratings = np.zeros((train_cust_n, train_movies_n))

    for i in range(len(train_df)):
        current = train_df.iloc[i]
        customer_idx = train_cid.get(current[1])
        movie_idx = train_mid.get(current[0])
        ratings[customer_idx, movie_idx] = current[2]
    
    return ratings

def construct_dict(df, column):
    # create a mapping from ID to an index (most records contain repeated IDs)
    col = df[column]
    col_sorted = col.sort_values()
    d = {}
    cursor = 0
    for val in col_sorted:
        if val not in d:
            d[val] = cursor
            cursor += 1
    return d

def preprocess_matrix(ratings, process_type=2):
    # create F_0 by:
    # type 1: demean each column
    # type 2: filling all empty spaces in a col with mean of the col
    f_ratings = ratings.copy()
    f_ratings_t = np.transpose(f_ratings)
    means = []
    for i in range(f_ratings_t.shape[0]):
        record = f_ratings_t[i]
        nonzeros = record[np.nonzero(record)]
        if len(nonzeros) > 0:
            mean = np.mean(nonzeros)
        else:
            mean = 0
        means.append(mean)

        if process_type == 1:
            record[np.nonzero(record)] = record[np.nonzero(record)] - mean
            f_ratings_t[i] = record

        elif process_type == 2:
            f_ratings_t[i] = np.where(record==0, mean, record)

    return np.transpose(f_ratings_t), means

def perform_svd_reconstruct(matrix, means, k=25, method=None, process_type=2):
    # use sparse for a sparse matrix(lot of zeros) eg. when F_0 uses type 1
    if method == 'sparse':
        U1, sigma, Vt = svds(f_ratings, k = k)
        sigma = np.diag(sigma)
        if process_type==1:
            preds = np.dot(np.dot(U1, sigma), Vt) + np.array(means)
        else:
            preds = np.dot(np.dot(U1, sigma), Vt)
    else:
        U, s, Vh = svd(f_ratings, full_matrices=False)
        temp1 = U [:, :k]
        temp2 = Vh [:k]
        temp3 = np.diag(s[:k])
        if process_type==1:
            preds = np.matmul(np.matmul(temp1, temp3), temp2) + np.array(means)
        else:
            preds = np.matmul(np.matmul(temp1, temp3), temp2)
    
    return preds

def predict(reconstructed_ratings, ratings, val_df, train_cid, train_mid, validate):
    pred = []
    gt = []
    for i in range(val_df.shape[0]):

        temp = val_df.iloc[i]
        cid = train_cid.get(temp[1])
        mid = train_mid.get(temp[0])

        if mid is not None:
            if cid is not None:
                rating_prediction = reconstructed_ratings[cid][mid]
            else:
                movie_r = ratings[:, mid]
                nonzeros = movie_r[np.nonzero(movie_r)]
                rating_prediction = np.mean(nonzeros)
        else:
            if cid is not None:
                movie_r = ratings[cid]
                nonzeros = movie_r[np.nonzero(movie_r)]
                rating_prediction = np.mean(nonzeros)
            else:
                rating_prediction = 3

        if temp[2] == 0:
            continue
        
        if validate:
            gt.append(temp[2])
    
        if rating_prediction > 5:
            rating_prediction = 5
        if rating_prediction <= 0.5:
            rating_prediction = 1

        pred.append(round(rating_prediction))
    
    return pred, gt


# main

# set validate to true if you want to perform validation, 
# set to false if you want to predict on test dataset
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

f_ratings, means = preprocess_matrix(ratings)

reconstructed_ratings = perform_svd_reconstruct(f_ratings, means)

if not validate:
    val_df_original = test_df.copy()

