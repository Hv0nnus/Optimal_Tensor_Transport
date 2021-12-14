import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import pickle
import networkx as nx

import gzip
import json
import os
import pandas as pd

pd.options.display.float_format = '{:,}'.format


# ------------BOOKS--------------------

def noise_sparsity(C, noise, sparsity):
    C[np.random.rand(*C.shape) > sparsity] = 0
    noisy_pair = np.random.rand(*C.shape) < noise
    C[(C == -1) & noisy_pair] = 1
    C[(C == 1) & noisy_pair] = -1
    return C


def df_to_numpy(path1="./Books/subdataset/books_2D.csv",
                path2="./Books/subdataset/books_info.csv",
                outpath="./Books/subdataset/C.pickle",
                noise=0, sparsity=1):
    np.random.seed(123456)
    small_df = pd.read_csv(path1)
    genre_df = pd.read_csv(path2)
    userId = np.array(small_df["userId"])
    movieId = np.array(small_df["movieId"])
    rating = np.array(small_df["rating"])
    label_genre = np.array(genre_df["genres"])
    movieId_reduce = np.array(genre_df["movieId"])
    # movieId_reduce define the order. label_genre and C[:, j] are order in the same way
    n_user = len(np.unique(userId))
    n_movie = len(movieId_reduce)

    C = np.empty((n_user, n_movie, n_movie))
    # Label_user define the order. The label returned and the C[i] are order in the same way.
    label_user = np.unique(userId)
    real_sparsity = 0
    for i, label_i in enumerate(label_user):
        rating_i = np.zeros(n_movie) + np.percentile(rating[userId == label_i], 33)
        for j, label_j in enumerate(movieId_reduce):
            if ((userId == label_i) & (movieId == label_j)).sum() >= 1:
                rating_i[j] = rating[(userId == label_i) & (movieId == label_j)]
            real_sparsity += 1

        C[i] = (2 * (rating_i[:, np.newaxis] >= rating_i[np.newaxis, :]) - 1) - \
               ((rating_i[:, np.newaxis] == rating_i[np.newaxis, :]) * 1)
        C[i] = noise_sparsity(C[i], noise=noise, sparsity=sparsity)

    real_sparsity /= len(label_user) * len(movieId_reduce)
    dict_pickle = {"C": [C],
                   "True_label": [label_user, label_genre],
                   "T_pos": [[0, 1, 1]],
                   "noise": noise,
                   "sparsity": sparsity,
                   "real_sparsity": real_sparsity}

    with open(outpath, 'wb') as handle:
        pickle.dump(dict_pickle, handle)


def assign_label(movies_df, df, genres, n_per_class_):
    movies, count = np.unique(df["movieId"], return_counts=True)
    movies_with_more_comments = count.argsort()[::-1]
    x_return = np.zeros(len(movies), dtype=int)
    movies_genre = np.array(movies_df["genres"])
    movies_Id = np.array(movies_df["movieId"])
    for m in range(len(movies)):
        pos_movie_m = np.where(movies_Id == movies[movies_with_more_comments[m]])
        assert len(pos_movie_m) == 1
        pos_movie_m = pos_movie_m[0]
        x = movies_genre[pos_movie_m][0]
        movies_genres = x.split("|")
        a = []
        for i in range(len(genres)):
            for j in range(len(genres[i])):
                if genres[i][j] in movies_genres:
                    a.append(i)
                    break
        if len(a) == 1 and n_per_class_[a[0]] > 0:
            n_per_class_[a[0]] -= 1
            x_return[pos_movie_m] = a[0]
        elif len(a) > 1 and n_per_class_[a[1]] > 0 and "War" in genres[1] and "Thriller" in genres[0]:
            #  Special case for War Western vs thriller
            n_per_class_[a[1]] -= 1
            x_return[pos_movie_m] = a[1]
        else:
            x_return[pos_movie_m] = -1
    return x_return, n_per_class_


def select_same_labels(df, movies_df,
                       genres=["Mystery_Thriller_Crime_Drama", "Fantasy_Sci-Fi"],
                       n_per_class=[10, 10]):
    movies_genre = [genre.split("_") for genre in genres]
    n_per_class_ = n_per_class.copy()

    new_list_genre, n_per_class_ = assign_label(movies_df=movies_df, df=df,
                                                genres=movies_genre, n_per_class_=n_per_class_)

    movies_df = movies_df.rename(columns={"genres": "init_genres"})
    movies_df.insert(2, "genres", new_list_genre, True)

    if sum(n_per_class_) > 0:
        print("Warning: Not all the points per classes are selected", n_per_class_)
    movies_df = movies_df.loc[movies_df["genres"].isin(range(len(genres)))]
    df = df.loc[df["movieId"].isin(movies_df["movieId"])]
    return df, movies_df, n_per_class_



def create_movies_dataset(path="./Movies/subdataset/", n_user=100, n_movie=100, n_per_class=[10, 10],
                          genres=["Mystery_Thriller_Crime_Drama", "Fantasy_Sci-Fi"],
                          name="", time_split=True, additional_path_save="", rdm_seed=123456):
    np.random.seed(rdm_seed)
    small_df = pd.read_csv(path + '../ratings' + '.csv')

    movies, count = np.unique(small_df["movieId"], return_counts=True)
    movies_with_more_comments = count.argsort()[-n_movie:][::-1]
    small_df = small_df.loc[small_df['movieId'].isin(movies[movies_with_more_comments])]

    movies_df = pd.read_csv(path + '../movies' + '.csv')
    movies_df = movies_df.loc[movies_df["movieId"].isin(movies[movies_with_more_comments])]

    # Chose the users that comment the most with respect to the movies chosen.
    users, count = np.unique(small_df["userId"], return_counts=True)
    users_with_more_comments = count.argsort()[-n_user:][::-1]
    small_df = small_df.loc[small_df['userId'].isin(users[users_with_more_comments])]

    if time_split:
        movies_title = np.array(movies_df["title"])
        movies_date = []
        for i in movies_title:
            try:
                movies_date.append(int(re.findall("\([0-9]+\)", i)[-1][1:-1]))
            except:
                movies_date.append(2018)
        movies_date = np.array(movies_date)
        movies_df["date"] = movies_date
        first_half = movies_date >= np.median(movies_date)
        first_half = movies_df.loc[first_half, "title"]
        movies_df_ = movies_df.loc[movies_df["title"].isin(first_half)]
        small_df_ = small_df.loc[small_df["movieId"].isin(movies_df_["movieId"])]

        small_df_, movies_df_, n_per_class_ = select_same_labels(small_df_, movies_df_,
                                                                 genres=genres,
                                                                 n_per_class=n_per_class)

        movies_df_.to_csv(path + additional_path_save + "movies_info" + name + "1.csv")
        small_df_.to_csv(path + additional_path_save + "movies_2D" + name + "1.csv")

        second_half = np.array([i for i in movies_title if i not in first_half])
        # second_half = movies_date < np.median(movies_date)
        # small_df["timestamp"] < np.median(small_df["timestamp"])
        movies_df_ = movies_df.loc[movies_df["title"].isin(second_half)]
        small_df_ = small_df.loc[small_df["movieId"].isin(movies_df_["movieId"])]
        # Chose a sub sample of movies by looking at the labels
        small_df_, movies_df_, n_per_class_ = select_same_labels(small_df_, movies_df_,
                                                                 genres=genres,
                                                                 n_per_class=n_per_class)
        movies_df_.to_csv(path + additional_path_save + "movies_info" + name + "2.csv")
        small_df_.to_csv(path + additional_path_save + "movies_2D" + name + "2.csv")
    else:
        # Chose a sub sample of movies by looking at the labels
        small_df, movies_df, n_per_class_ = select_same_labels(small_df, movies_df,
                                                               genres=genres,
                                                               n_per_class=n_per_class)
        movies_df.to_csv(path + additional_path_save + "movies_info" + name + ".csv")
        small_df.to_csv(path + additional_path_save + "movies_2D" + name + ".csv")


def create_dataset(rdm_seed):
    genres = [["Thriller_Crime_Drama", "Fantasy_Sci-Fi"],
              ["Thriller_Crime_Drama", "Children's_Animation"],
              ["Thriller_Crime_Drama", "War_Western"],
              ["Fantasy_Sci-Fi", "Children's_Animation"],
              ["Fantasy_Sci-Fi", "War_Western"],
              ["Children's_Animation", "War_Western"]
              ]

    names = ["_" + genre[0][0] + "_" + genre[1][0] for genre in genres]
    n_movies = [3000, 3000, 5000, 5000, 5000, 5000]
    for i, name in enumerate(names):
        create_movies_dataset(path="./Movies/subdataset/", n_user=100, n_movie=n_movies[i], n_per_class=[100, 100],
                              name=name, genres=genres[i], rdm_seed=rdm_seed)
        noise = 0
        sparsity = 1
        df_to_numpy(path1="./Movies/subdataset/movies_2D" + name + "1.csv",
                    path2="./Movies/subdataset/movies_info" + name + "1.csv",
                    outpath="./Movies/subdataset/C" + name + "1.pickle",
                    noise=noise, sparsity=sparsity)
        df_to_numpy(path1="./Movies/subdataset/movies_2D" + name + "2.csv",
                    path2="./Movies/subdataset/movies_info" + name + "2.csv",
                    outpath="./Movies/subdataset/C" + name + "2.pickle",
                    noise=noise, sparsity=sparsity)
