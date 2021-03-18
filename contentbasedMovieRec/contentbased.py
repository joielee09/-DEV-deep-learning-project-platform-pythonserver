# -*- coding: utf-8 -*-

# Copyright permitted

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, sys 


df_ratings = pd.read_csv('./contentbasedMovieRec/ratings.csv')
df_ratings.drop(['timestamp'], axis=1, inplace=True)


df_movies = pd.read_csv('./contentbasedMovieRec/movies.csv')

rows = []                               # row = [user_id, movie_id, rating]
user_id = 800
rows.append([user_id, 1, 4])        # movie     1: Toy Story(1995)
rows.append([user_id, 4896, 4])     # movie  4896: Harry Potter and the Socerer's Stone 
rows.append([user_id, 5816, 5])     # movie  5896: Harry Potter and the Chamber of Secrets
rows.append([user_id, 69844, 5])    # movie 69844: Harry Potter and the Half-Blood Prince(2009)
rows.append([user_id, 12, 1])       # movie    12: Dracula: Dead and Loving It(1995)
rows.append([user_id, 177, 1])      # movie   177: Lord of Illusions(1995)

for row in rows:
    df_ratings = df_ratings.append(pd.Series(row, index=df_ratings.columns), ignore_index=True)

n_users = df_ratings.userId.unique().shape[0]
n_items = df_ratings.movieId.unique().shape[0]

movie_rate = dict()

for row in df_ratings.itertuples(index = False):
    user_id, movie_id, rate = row
    if movie_id not in movie_rate:
        movie_rate[movie_id] = [0, 0]
    movie_rate[movie_id][1] += 1

for key, value in movie_rate.items():
    value1 = value[0] / value[1]
    movie_rate[key] = [round(value1, 3),value[1]]


user_dict = dict()
movie_dict = dict()
user_idx = 0
movie_idx = 0
ratings = np.zeros((n_users, n_items))
for row in df_ratings.itertuples(index=False):
    user_id, movie_id, _ = row
    if user_id not in user_dict:
        user_dict[user_id] = user_idx
        user_idx += 1
    if movie_id not in movie_dict:
        movie_dict[movie_id] = movie_idx
        movie_idx += 1
    ratings[user_dict[user_id], movie_dict[movie_id]] = row[2]
user_idx_to_id = {v: k for k, v in user_dict.items()}

movie_idx_to_name=dict()
movie_idx_to_genre=dict()
for row in df_movies.itertuples(index=False):
    movie_id, movie_name, movie_genre = row
    if movie_id not in movie_dict:
        continue
    movie_idx_to_name[movie_dict[movie_id]] = movie_name 
    movie_idx_to_genre[movie_dict[movie_id]] = movie_genre

df_movies['genres'] = df_movies['genres'].apply(lambda x : x.split('|')).apply(lambda x : " ".join(x))

df_movies

rates = dict()
rates['movieId'] = []
rates['score'] = []
rates['count'] = []
for key, value in movie_rate.items():
    rates['movieId'].append(key)
    rates['score'].append(value[0])
    rates['count'].append(value[1])

scores = pd.DataFrame(rates)
scores

df_movies = pd.merge(df_movies, scores, on='movieId')

df_movies.head(4)

tmp_m = df_movies['count'].quantile(0.89)
tmp_m

tmp_data = df_movies.copy().loc[df_movies['count'] >= tmp_m]
tmp_data.shape

del tmp_data

m = df_movies['count'].quantile(0.9)
data = df_movies.loc[df_movies['count'] >= m]

df_movies.head()

C = df_movies['score'].mean()

def weighted_rating(x, m=m, C=C):
    v = x['count']
    R = x['score']
    
    return ( v / (v+m) * R ) + (m / (m + v) * C)

df_movies['weighted_score'] = df_movies.apply(weighted_rating, axis = 1)
df_movies.head(4)

count_vector = CountVectorizer(ngram_range=(1, 3))
count_vector

c_vector_genres = count_vector.fit_transform(df_movies['genres'])
c_vector_genres

c_vector_genres.shape

gerne_c_sim = cosine_similarity(c_vector_genres, c_vector_genres).argsort()[:, ::-1]

gerne_c_sim.shape

def get_recommend_movie_list(df, movie_title, top=30):
    target_movie_index = df[df['title'] == movie_title].index.values
    sim_index = gerne_c_sim[target_movie_index, :top].reshape(-1)
    sim_index = sim_index[sim_index != target_movie_index]
    result = df.iloc[sim_index].sort_values('weighted_score', ascending=False)[:20]
    return result

get_recommend_movie_list(df_movies, movie_title='Toy Story (1995)')

import requests
from urllib.request import urlopen
from PIL import Image

def movie_poster(titles):
    data_URL = 'http://www.omdbapi.com/?i=tt3896198&apikey=f9cdaffd'
    
    fig, axes = plt.subplots(2, 10, figsize=(30,9))
    
    for i, ax in enumerate(axes.flatten()):
        w_title = titles[i].strip().split()
        params = {
            's':titles[i],
            'type':'movie',
            'y':''    
        }
        response = requests.get(data_URL,params=params).json()
        
        if response["Response"] == 'True':
            poster_URL = response["Search"][0]["Poster"]
            img = Image.open(urlopen(poster_URL))
            ax.imshow(img)
            
        ax.axis("off")
        if len(w_title) >= 10:
            ax.set_title(f"{i+1}. {' '.join(w_title[:5])}\n{' '.join(w_title[5:10])}\n{' '.join(w_title[10:])}", fontsize=10)
        elif len(w_title) >= 5:
            ax.set_title(f"{i+1}. {' '.join(w_title[:5])}\n{' '.join(w_title[5:])}", fontsize=10)
        else:
            ax.set_title(f"{i+1}. {titles[i]}", fontsize=10)
        
    plt.show()

def moviepredict(title):
    rec2 = get_recommend_movie_list(df_movies, movie_title=title)
    rec2 = rec2['title'].apply(lambda x : x.split(' (')[0])
    return rec2.to_json(orient="split")

