import streamlit as st
import pandas as pd
import numpy as np
import pickle
from surprise import SVD, Dataset, Reader

# Load dữ liệu từ các tệp pickle
smd = pd.read_pickle('processed_data/smd.pkl')
indices = pd.read_pickle('processed_data/indices.pkl')
id_map = pd.read_pickle('processed_data/id_map.pkl')
indices_map = pd.read_pickle('processed_data/indices_map.pkl')

# Load ma trận cosine_sim từ tệp numpy
cosine_sim = np.load('processed_data/cosine_sim.npy')

# Load mô hình SVD (giả định bạn đã lưu mô hình trước đó)
with open('processed_data/svd_model.pkl', 'rb') as f:
    svd = pickle.load(f)

def hybrid(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    movie_id = id_map.loc[title]['movieId']
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)

# Streamlit app
st.title("Movie Recommendation System")

user_id = st.number_input("Enter User ID", min_value=1, step=1)
movie_title = st.text_input("Enter Movie Title")

if st.button("Recommend"):
    if movie_title in indices:
        recommendations = hybrid(user_id, movie_title)
        st.write(recommendations)
    else:
        st.write("Movie title not found in the dataset.")
