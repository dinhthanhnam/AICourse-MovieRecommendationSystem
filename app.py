import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests

# Load dữ liệu từ các tệp pickle
smd = pd.read_pickle('AI_Dump/smd.pkl')
indices = pd.read_pickle('AI_Dump/indices.pkl')
id_map = pd.read_pickle('AI_Dump/id_map.pkl')
indices_map = pd.read_pickle('AI_Dump/indices_map.pkl')

# Load ma trận cosine_sim từ tệp numpy
cosine_sim = np.load('AI_Dump/cosine_sim.npy')

# Load mô hình SVD
with open('AI_Dump/svd_model.pkl', 'rb') as f:
    svd = pickle.load(f)

# Đọc dữ liệu đánh giá từ tệp CSV
ratings = pd.read_csv('input_data/ratings_small.csv')

# Lấy danh sách user ID duy nhất
user_ids = ratings['userId'].unique()

# API key của bạn từ TMDb
TMDB_API_KEY = 'f3a8f89e1d0435429b474be6a159709a'

# Hàm lấy URL của ảnh phim từ TMDb
def get_movie_poster_url(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path', None)
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

# Gợi ý phim kết hợp (userId và title)
def hybrid(userId, title):
    idx = indices[title]
    print("index:", idx)
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False).head(10)
    movies['poster_url'] = movies['id'].apply(get_movie_poster_url)
    return movies

# Streamlit GUI
st.title("Hệ thống đề xuất phim")

# Chọn user ID
selected_user_id = st.selectbox("Chọn userid: ", user_ids)

# Chọn tiêu đề phim
movie_titles = smd['title'].values
selected_movie_title = st.selectbox("Chọn tựa đề phim: ", movie_titles)

if st.button("Đề xuất"):
    recommendations = hybrid(selected_user_id, selected_movie_title)
    
    st.write(f"Đề xuất cho phim {selected_movie_title}:")
    
    # Số cột bạn muốn hiển thị
    num_cols = 5
    # Tạo danh sách các cột
    cols = st.columns(num_cols)
    # Tính toán số phim mỗi cột
    num_movies_per_col = int(np.ceil(len(recommendations) / num_cols))
    
    for i, col in enumerate(cols):
        with col:
            start_idx = i * num_movies_per_col
            end_idx = min((i + 1) * num_movies_per_col, len(recommendations))
            for idx in range(start_idx, end_idx):
                row = recommendations.iloc[idx]
                st.write(f"**{row['title']}**")
                st.write(f"Năm: {row['year']}")
                st.write(f"Est: {row['est']:.2f}")
                if row['poster_url']:
                    st.image(row['poster_url'])
                else:
                    st.write("Không tìm thấy poster.")
