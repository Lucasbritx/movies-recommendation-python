import numpy as np
import pandas as pd
import re
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# LOAD DATASET

dataset_path = os.path.join(os.path.dirname(__file__), './venv/the-movies-dataset/ratings_small.csv')
data = pd.read_csv(dataset_path)


print(f'Shape of Small data : {data.shape}')

data

len(set(data['userId']))
print(f'There are {len(set(data["movieId"]))} film available')

pivot_matrix = data.pivot(index='userId', columns='movieId', values='rating')   # CHANGE TO SPARSE MATRIX
pivot_matrix = pivot_matrix.fillna(0)  # FILL MISSING VALUES WITH 0

pivot_matrix

# PREDICT MOVIE THAT HAS NOT BEEN RATED BY USERS (USER-BASED COLLABORATIVE FILTERING)

def user_based_recommendation(target_user_id, pivot_matrix, similarity_df, n_recommendations):

    # EXAMPLE: PREDICT RATED MOVIE FOR userId = 1
    users_rating = pivot_matrix.loc[target_user_id]

    # TAKE MOVIE THAT HAVE BEEN RATED BY USERS (RATING > 0)
    rated_items = users_rating[users_rating > 0].index.tolist()


    # CALCULATE RECOMMENDATION SCORE FOR MOVIES THAT HAVE NOT BEEN RATED BY USERS
    recommendations = {}
    for movie in users_rating[users_rating == 0].index:  # ITERATED FOR EACH UNRATED MOVIE

        similar_score = 0  # TO CALCULATE THE SIMILARITY SCORE FOR EACH MOVIE THAT HASN'T BEEN RATED
        similarity_sum = 0
    
        # ITERATE THROUGH ALL USERS TO FIND SIMILARITY
        for user_id in user_similarity_df.index:

            if user_id != target_user_id:  # SKIP THE TARGET THEMSELF
                similarity_score = user_similarity_df[target_user_id][user_id]  # GET SIMILARITY SCORES BETWEEN CURRENT USER AND OTHER USER
            
                # ONLY CONSIDER USERS WHO HAVE RATED THIS MOVIE
                if pivot_matrix.loc[user_id, movie] > 0:  # CHECK IF USER HAS RATED THIS MOVIE

                    similar_score += similarity_score * pivot_matrix.loc[user_id, movie]  # CALCULATE THE SUM OF USERS SIMILARITY + RATED MOVIE FOR EACH USER
                    similarity_sum += abs(similarity_score)  # TOTAL SIMILARITY WEIGHT
    
        # TO AVOID DIVISION BY ZERO
        if similarity_sum > 0:
            recommendations[movie] = similar_score / similarity_sum  # NORMALIZED RECOMMENDATION SCORE
        else:
            recommendations[movie] = 0


    # SORT RECOMMENDATIONS IN DESCENDING ORDER OF PREDICTED RATING
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

    # PRINT TOP RECOMMENDATIONS (TOP N RECOMMENDATIONS)
    top_recommendations = sorted_recommendations[:n_recommendations]

    return top_recommendations

# BUILDING ITEM-BASED COLLABORATIVE FILTERING USING COSINE SIMILARITY

item_similarity = cosine_similarity(pivot_matrix.T)

item_similarity_df = pd.DataFrame(item_similarity, index= pivot_matrix.columns, columns= pivot_matrix.columns)

# DISPLAY SIMILARITY FOR EACH OTHER MOVIE
item_similarity_df  


# FUNCTION TO PREDICT MOVIE THAT HAVE NOT RATED BY USERS

def item_based_recommendation(user_id, pivot_matrix, similarity_df, n_recommendations):

    # GET TARGET USERS
    users_rating = pivot_matrix.loc[user_id]

    # TAKE MOVIE THAT HAVE BEEN RATED BY USERS (RATING > 0)
    rated_items = users_rating[users_rating > 0].index.tolist()

    # CALCULATE RECOMMENDATION SCORE FOR MOVIES THAT HAVE NOT BEEN RATED BY USERS
    recommendations = {}
    for movie in item_similarity_df.index:    # ITERATE FOR EACH MOVIE

        similar_score = 0   # TO CALCULATE SIMILAR SCORE FOR EACH MOVIE THAT HASN'T RATED BY USER
        similarity_sum = 0
    
        # ITERATE ALL NON RATED MOVIES BY USER
        if movie not in rated_items:
        
            # CALCULATE SIMILARITY SCORES BASED ON ITEMS THAT HAVE BEEN RATED BY USERS
            for rated in rated_items:
                movie_similarity_score = item_similarity_df[movie][rated]       # GET SIMILARITY BETWEEN UNRATED MOVIE AND RATED MOVIE
                similar_score += movie_similarity_score * users_rating[rated]   # CALCULATE BETWEEN NON RATED MOVIE AND RATED MOVIE

                similarity_sum += abs(movie_similarity_score)  # TOTAL 

        # TO AVOID DIVISION BY ZERO
        if similarity_sum > 0:
            recommendations[movie] = similar_score / similarity_sum   # NORMALIZATION
        else:
            recommendations[movie] = 0
        
    # SORT RECOMMENDATIONS IN DESCENDING ORDER OF PREDICTED RATING
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

    # PRINT TOP RECOMMENDATIONS (TOP N RECOMMENDATIONS)
    top_recommendations = sorted_recommendations[:n_recommendations]

    return top_recommendations

    # PREDICT RECOMMENDATIONS FROM SPECIFIC USERS

target_id = 1
n = 10
result = item_based_recommendation(target_id, pivot_matrix, item_similarity_df, n)  # SUPPOSE WE WANT TO PREDICT THE TOP 10 MOVIES THAT WILL BE RECOMMENDED FOR USER 1

print(f"Top {n} movie recommendations for user {target_id}:")
for movie, score in result:
    print(f"Movie: {movie}, Predicted Rating: {score}")

    # PREDICT RECOMMENDATIONS FROM SPECIFIC USERS

target_id = 671
n = 20
result = item_based_recommendation(target_id, pivot_matrix, item_similarity_df, n)  # SUPPOSE WE WANT TO PREDICT THE TOP 10 MOVIES THAT WILL BE RECOMMENDED FOR USER 1

print(f"Top {n} movie recommendations for user {target_id}:")
for movie, score in result:
    print(f"Movie: {movie}, Predicted Rating: {score}")

# USER BASED COLLABORATIVE FILTERING USING PEARSON CORRELATION
user_similarity_df = pivot_matrix.T.corr(method='pearson')

# DISPLAY INFORMATION
print(f'Type  : {type(user_similarity_df)}')
print(f'Shape : {len(user_similarity_df)}')

# CONVERT TO DATAFRAME
user_similarity_df


# RECOMMENDATIONS 10 THAT USERS LIKE

target_id = 1
n = 10
result = user_based_recommendation(target_id, pivot_matrix, item_similarity_df, n)  # SUPPOSE WE WANT TO PREDICT THE TOP 10 MOVIES THAT WILL BE RECOMMENDED FOR USER 1

print(f"Top {n} movie recommendations for user {target_id}:")
for movie, score in result:
    print(f"Movie: {movie}, Predicted Rating: {score}")

    # PREDICT RECOMMENDATIONS FROM SPECIFIC USERS

target_id = 230
n = 10
result = user_based_recommendation(target_id, pivot_matrix, item_similarity_df, n)  # SUPPOSE WE WANT TO PREDICT THE TOP 10 MOVIES THAT WILL BE RECOMMENDED FOR USER 1

print(f"Top {n} movie recommendations for user {target_id}:")
for movie, score in result:
    print(f"Movie: {movie}, Predicted Rating: {score}")