import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

from surprise import SVD
from surprise import Dataset
from surprise.model_selection import GridSearchCV
from surprise import Reader

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

import streamlit as st

import pickle
import os


nltk.download('stopwords')

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

ratings.head()

movies.head(10)

ratings_array = ratings['rating'].unique()
max_rating = np.amax(ratings_array)
min_rating = np.amin(ratings_array)
print(ratings_array)

movie_map = pd.Series(movies.movieId.values, index=movies.title).to_dict()
reverse_movie_map = {v: k for k, v in movie_map.items()}
movieId_to_index_map = pd.Series(
    movies.index.values, index=movies.movieId).to_dict()
movieId_all_array = movies['movieId'].unique()


def get_movieId(movie_name):
    """
    return the movieId which is corresponding to the movie name

    Parameters
    ----------
    movie_name: string, the name of the movie w/ or w/o the year

    Return
    ------
    the movieId
    """

    # If luckily the movie name is 100% equal to a name writen in the database,
    # then return the id corresponding to the name.
    # Or...we need to consider the similarity between strings
    if (movie_name in movie_map):
      return movie_map[movie_name]
    else:
      similar = []
      for title, movie_id in movie_map.items():
        ratio = fuzz.ratio(title.lower(), movie_name.lower())
        if (ratio >= 60):
          similar.append((title, movie_id, ratio))
      if (len(similar) == 0):
        print("Oh! This movie does not exist in the database.")
      else:
        match_item = sorted(similar, key=lambda x: x[2])[::-1]
        print("The matched item might be:",
              match_item[0][0], ", ratio=", match_item[0][2])
        return match_item[0][1]


#Build up the content-based filtering algorithm with pairwise approach in TF-IDF vector space


def tokenizer(text):
  torkenized = [PorterStemmer().stem(word).lower() for word in text.split(
      '|') if word not in stopwords.words('english')]
  return torkenized


cos_sim = 0
if (os.path.exists("./model/similarity.pkl")):
    cos_sim = pickle.load(open('model/similarity.pkl', 'rb'))
else:
	print("Model doesnt exist")
	tfid = TfidfVectorizer(analyzer='word', tokenizer=tokenizer)

	tfidf_matrix = tfid.fit_transform(movies['genres'])

	cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
	pickle.dump(cos_sim, open('model/similarity.pkl', 'wb'))
	print(tfidf_matrix.shape)
	print(cos_sim.shape)
	print(movies.shape)

#Build up the Singular Value Decomposition (SVD) matrix factorization model in collaborative filtering algorithm
features = ['userId', 'movieId', 'rating']
best_params = 0
if (os.path.exists("./model/svd.pkl")):
  best_params = pickle.load(open('model/svd.pkl', 'rb'))
  print(best_params)
else:
	reader = Reader(rating_scale=(min_rating, max_rating))
	data = Dataset.load_from_df(ratings[features], reader)
	param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
               'reg_all': [0.4, 0.6]}
	gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
	gs.fit(data)
	print(gs.best_score['rmse'])
	print(gs.best_params['rmse'])

	best_params = gs.best_params['rmse']
	pickle.dump(best_params, open('model/svd.pkl', 'wb'))


def get_rating_from_prediction(prediction, ratings_array):
    """
    return the closest rating number to the prediction value

    Parameters
    ----------
    prediction: float, the prediction value from the model

    ratings_array: the 1D array of the discrete rating number

    Return
    ------
    the rating number corresponding to the prediction value
    """
    rating = ratings_array[np.argmin(
        [np.abs(item - prediction) for item in ratings_array])]
    return rating


# prediction = model_svd.predict(1, 1)
# print("rating", ratings[(ratings.userId == 1)
#       & (ratings.movieId == 1)]['rating'])
# print("prediction", prediction.est)

#Make movie recommendation (item-based)


def make_recommendation_item_based(similarity_matrix, movieId_all_array, ratings_data, id_to_movie_map, movieId_to_index_map, fav_movie_list, n_recommendations, userId=-99):
    """
    return top n movie recommendation based on user's input list of favorite movies
    Currently, fav_movie_list only support one input favorate movie

    Parameters
    ----------
    similarity_matrix: 2d array, the pairwise similarity matrix

    movieId_all_array: 1d array, the array of all movie Id

    ratings_data: ratings data

    id_to_movie_map: the map from movieId to movie title

    movieId_to_index_map: the map from movieId to the index of the movie dataframe

    fav_movie_list: list, user's list of favorite movies

    n_recommendations: int, top n recommendations

    userId: int optional (default=-99), the user Id
            if userId = -99, the new user will be created
            if userId = -1, the latest inserted user is chosen

    Return
    ------
    list of top n movie recommendations

    """

    if (userId == -99):
      userId = np.amax(ratings_data['userId'].unique()) + 1
    elif (userId == -1):
      userId = np.amax(ratings_data['userId'].unique())

    movieId_list = []
    for movie_name in fav_movie_list:
      movieId_list.append(get_movieId(movie_name))

    # Get the movie id which corresponding to the movie the user didn't watch before
    movieId_user_exist = list(
        ratings_data[ratings_data.userId == userId]['movieId'].unique())
    movieId_user_exist = movieId_user_exist + movieId_list
    movieId_input = []
    for movieId in movieId_all_array:
      if (movieId not in movieId_user_exist):
         movieId_input.append(movieId)

    index = movieId_to_index_map[movieId_list[0]]
    cos_sim_scores = list(enumerate(similarity_matrix[index]))
    cos_sim_scores = sorted(cos_sim_scores, key=lambda x: x[1], reverse=True)

    topn_movieIndex = []
    icount = 0
    for i in range(len(cos_sim_scores)):
      if(cos_sim_scores[i][0] in [movieId_to_index_map[ids] for ids in movieId_input]):
        icount += 1
        topn_movieIndex.append(cos_sim_scores[i][0])
        if(icount == n_recommendations):
          break

    topn_movie = [movies.loc[index].title for index in topn_movieIndex]
    return topn_movie


#Make movie recommendation (user-based)


def make_recommendation_user_based(best_model_params, movieId_all_array, ratings_data, id_to_movie_map,
                                   fav_movie_list, n_recommendations, userId=-99):
    """
    return top n movie recommendation based on user's input list of favorite movies
    Currently, fav_movie_list only support one input favorate movie


    Parameters
    ----------
    best_model_params: dict, {'iterations': iter, 'rank': rank, 'lambda_': reg}

    movieId_all_array: the array of all movie Id

    ratings_data: ratings data

    id_to_movie_map: the map from movieId to movie title

    fav_movie_list: list, user's list of favorite movies

    n_recommendations: int, top n recommendations

    userId: int optional (default=-99), the user Id
            if userId = -99, the new user will be created
            if userId = -1, the latest inserted user is chosen

    Return
    ------
    list of top n movie recommendations
    """

    movieId_list = []
    for movie_name in fav_movie_list:
      movieId_list.append(get_movieId(movie_name))

    if (userId == -99):
      userId = np.amax(ratings_data['userId'].unique()) + 1
    elif (userId == -1):
      userId = np.amax(ratings_data['userId'].unique())

    ratings_array = ratings['rating'].unique()
    max_rating = np.amax(ratings_array)
    min_rating = np.amin(ratings_array)

    # create the new row which corresponding to the input data
    user_rows = [[userId, movieId, max_rating] for movieId in movieId_list]
    df = pd.DataFrame(user_rows, columns=['userId', 'movieId', 'rating'])
    train_data = pd.concat([ratings_data, df], ignore_index=True, sort=False)

    # Get the movie id which corresponding to the movie the user didn't watch before
    movieId_user_exist = train_data[train_data.userId ==
                                    userId]['movieId'].unique()
    movieId_input = []
    for movieId in movieId_all_array:
      if (movieId not in movieId_user_exist):
         movieId_input.append(movieId)

    reader = Reader(rating_scale=(min_rating, max_rating))

    data = Dataset.load_from_df(train_data, reader)

    model = SVD(**best_model_params)
    model.fit(data.build_full_trainset())

    predictions = []
    for movieId in movieId_input:
      predictions.append(model.predict(userId, movieId))

    sort_index = sorted(range(len(predictions)),
                        key=lambda k: predictions[k].est, reverse=True)
    topn_predictions = [predictions[i].est for i in sort_index[0:min(
        n_recommendations, len(predictions))]]
    topn_movieIds = [movieId_input[i]
                     for i in sort_index[0:min(n_recommendations, len(predictions))]]
    topn_rating = [get_rating_from_prediction(
        pre, ratings_array) for pre in topn_predictions]

    topn_movie = [id_to_movie_map[ids] for ids in topn_movieIds]
    return topn_movie


#Make a recommendation!
def display(data):
	text = ""
	i = 1
	for row in data:
		text += str(i) + ". " + row + "\n"
		i += 1
	return text

st.title("Movie Recommendation System")
movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Enter a Movie",
    movie_list
)

my_favorite_movies = []
my_favorite_movies.insert(0, selected_movie)

# get recommends
n_recommendations = 10
if (st.button("Search")):
	res1 = []
	res2 = []
	recommends_item_based = make_recommendation_item_based(
		similarity_matrix=cos_sim,
		movieId_all_array=movieId_all_array,
		ratings_data=ratings[features],
		id_to_movie_map=reverse_movie_map,
		movieId_to_index_map=movieId_to_index_map,
		fav_movie_list=my_favorite_movies,
		n_recommendations=n_recommendations)

	recommends_user_based = make_recommendation_user_based(
		best_model_params=best_params,
		movieId_all_array=movieId_all_array,
		ratings_data=ratings[features],
		id_to_movie_map=reverse_movie_map,
		fav_movie_list=my_favorite_movies,
		n_recommendations=n_recommendations)

	print("-------------Search based on item's content similarity--------------------")
	print('The movies similar to', my_favorite_movies, ':')
	for i, title in enumerate(recommends_item_based):
		print(i+1, title)
		res1.append(title)
	if(len(recommends_item_based) < n_recommendations):
		print("Sadly, we couldn't offer so many recommendations :(")

	print("--------------Search based on similarity between user's preference--------------------------------------")
	print('The users like', my_favorite_movies, 'also like:')
	for i, title in enumerate(recommends_user_based):
		print(i+1, title)
		res2.append(title)
	if(len(recommends_user_based) < n_recommendations):
		print("Sadly, we couldn't offer so many recommendations :(")

	st.subheader("Search based on item's content similarity")
	st.text(display(res1))
	st.subheader("Search based on similarity between user's preference")
	st.text(display(res2))