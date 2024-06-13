import pandas as pd
import numpy as np
import requests 
import pickle

smd = pd.read_pickle('AI_Dump/smd.pkl')
indices = pd.read_pickle('AI_Dump/indices.pkl')
id_map = pd.read_pickle('AI_Dump/id_map.pkl')
indices_map = pd.read_pickle('AI_Dump/indices_map.pkl')
#load ma trận cosine sime và user_ids
user_ids = np.load('AI_Dump/user_ids.npy')
cosine_sim = np.load('AI_Dump/cosine_sim.npy')
# Load mô hình SVD
with open('AI_Dump/svd_model.pkl', 'rb') as f:
    svd = pickle.load(f)

ratings = pd.read_csv('input_data/ratings_small.csv')
movie_dict = id_map['id'].to_dict()

TMDB_API_KEY ='f3a8f89e1d0435429b474be6a159709a'

def get_movie_titles(movie_id):
    try:
        titles = id_map[id_map['id'] == movie_id].index.tolist()
        return titles
    except IndexError:
        return []
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
def hybrid(userId, movie_id):
    idx = indices_map.index.get_loc(movie_id)
    print("index:", idx) 
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:16]
    movie_indices = [i[0] for i in sim_scores]
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False).head(10)
    movies['poster_url'] = movies['id'].apply(get_movie_poster_url)
    return movies

def personalized_movies_recommendations_v1(user_id):
    user_ratings = ratings[ratings['userId'] == user_id]
    user_movie_ids = user_ratings['movieId'].tolist()
    
    user_tmdb_ids = []
    for movie_id in user_movie_ids:
        movie_data = id_map[id_map['movieId'] == movie_id]
        if not movie_data.empty:
            tmdb_id = movie_data.iloc[0]['id']
            user_tmdb_ids.append(tmdb_id)
    
    unique_tmdb_ids = list(set(user_tmdb_ids))
    recommended_movies = pd.DataFrame(columns=['title', 'vote_count', 'vote_average', 'year', 'id'])
    for tmdb_id in unique_tmdb_ids:
        similar_movies = hybrid(user_id, tmdb_id)
        recommended_movies = pd.concat([recommended_movies, similar_movies])
    recommended_movies = recommended_movies.drop_duplicates(subset= 'id').sort_values('est', ascending=False).head(15)
    
    return recommended_movies

vote_counts = smd[smd['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = smd[smd['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
m = vote_counts.quantile(0.60)

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

def improved_recommendations(movie_id):
    print(movie_id)
    idx = indices_map.index.get_loc(movie_id)
    print(idx)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & 
                       (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified

def personalized_movies_recommendations_v2(user_id):
    user_ratings = ratings[ratings['userId'] == user_id]
    user_movie_ids = user_ratings['movieId'].tolist()
    unique_movie_ids = list(set(user_movie_ids))
    
    user_tmdb_ids = []
    for movie_id in unique_movie_ids:
        movie_data = id_map[id_map['movieId'] == movie_id]
        if not movie_data.empty:
            tmdb_id = movie_data.iloc[0]['id']
            user_tmdb_ids.append(tmdb_id)
    
    # Lấy ra danh sách phim phù hợp bằng improved_recommendations
    recommended_movies_list = []
    for movie_id in user_tmdb_ids:
        recommended_movies = improved_recommendations(movie_id)
        recommended_movies_list.append(recommended_movies)
    
    # Kết hợp danh sách phim phù hợp từ tất cả các phim người dùng đã xem
    all_recommended_movies = pd.concat(recommended_movies_list)
    all_recommended_movies = all_recommended_movies.drop_duplicates(subset=['id'])

    # Áp dụng SVD để tìm ra 15 phim tốt nhất
    all_recommended_movies['est'] = all_recommended_movies['id'].apply(lambda x: svd.predict(user_id, indices_map.loc[x]['movieId']).est)
    top_recommended_movies = all_recommended_movies.sort_values('est', ascending=False).head(15)
    top_recommended_movies['poster_url'] = top_recommended_movies['id'].apply(get_movie_poster_url)
    
    return top_recommended_movies

# def user_input_ratings_recommendations(user_ratings):
#     user_ratings_df = pd.DataFrame(list(user_ratings.items()), columns=['movieId', 'rating'])
#     user_movie_ids = user_ratings_df['movieId'].tolist()
#     recommended_movies_list = []
#     for movie_id in user_movie_ids:
#         recommended_movies = improved_recommendations(movie_id)
#         recommended_movies_list.append(recommended_movies)

#     all_recommended_movies = pd.concat(recommended_movies_list)
#     all_recommended_movies = all_recommended_movies.drop_duplicates(subset=['id'])
#     all_recommended_movies = all_recommended_movies.rename(columns={'wr': 'est'})
#     top_recommended_movies = all_recommended_movies.sort_values('est', ascending=False).head(15)
#     top_recommended_movies['poster_url'] = top_recommended_movies['id'].apply(get_movie_poster_url)
    
#     return top_recommended_movies









