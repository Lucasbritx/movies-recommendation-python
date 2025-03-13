import pandas as pd

# Carregar dados do MovieLens (100k)
movies = pd.read_csv("https://grouplens.org/datasets/movielens/32m/ml-25m/movies.csv")

print(movies.head())