import streamlit as st
from GUIHelper import hybrid, personalized_movies_recommendations_v1, personalized_movies_recommendations_v2, user_input_ratings_recommendations, movie_dict

def truncate_title_with_tooltip(title, max_length=13):
    if len(title) > max_length:
        truncated_title = title[:max_length] + '...'
        return f'<span title="{title}">{truncated_title}</span>'
    return title

def display_recommendations(recommendations, header):
    st.write(f"## {header}")
    
    # Display recommendations in 2 rows, each row containing 5 movies
    for i in range(3):
        row_recommendations = recommendations.iloc[i * 5 : (i + 1) * 5]
        cols = st.columns(5)
        for j, (_, row) in enumerate(row_recommendations.iterrows()):
            col = cols[j]
            with col:
                truncated_title_with_tooltip = truncate_title_with_tooltip(row['title'])
                st.markdown(truncated_title_with_tooltip, unsafe_allow_html=True)
                st.write(f"Đánh giá: {row['est']:.2f}")
                if row['poster_url']:
                    st.image(row['poster_url'], width=100)
                else:
                    st.write("Không tìm thấy ảnh.")

st.title('Hệ Thống Gợi Ý Phim')

# Add selectbox to choose between hybrid and personalized recommendations
recommendation_type = st.selectbox('Chọn loại gợi ý:', ['Hybrid', 'Cá nhân hoá V1','Cá nhân hoá V2', 'Đánh giá tự nhập'])

# Display input options based on selected recommendation type
if recommendation_type == 'Hybrid':
    selected_movie = st.selectbox('Chọn một bộ phim:', list(movie_dict.keys()))
    user_id = st.number_input('Nhập User ID:', min_value=1, key="user_id_hybrid_input")
    if st.button('Đề xuất'):
        movie_id = movie_dict[selected_movie]
        recommendations = hybrid(user_id, movie_id)
        display_recommendations(recommendations, "Phim đề xuất")

elif recommendation_type == 'Cá nhân hoá V1':
    user_id_personalized = st.number_input('Nhập User ID:', min_value=1, key="user_id_personalized_input")
    if st.button('Đề xuất'):
        recommendations_personalized = personalized_movies_recommendations_v1(user_id_personalized)
        display_recommendations(recommendations_personalized, f"Phim đề xuất cho người dùng {user_id_personalized}")

elif recommendation_type == 'Cá nhân hoá V2':
    user_id_personalized = st.number_input('Nhập User ID:', min_value=1, key="user_id_personalized_input")
    if st.button('Đề xuất'):
        recommendations_personalized = personalized_movies_recommendations_v2(user_id_personalized)
        display_recommendations(recommendations_personalized, f"Phim đề xuất cho người dùng {user_id_personalized}")

# elif recommendation_type == 'Đánh giá tự nhập':
#     st.write('Nhập đánh giá của bạn:')
#     user_ratings = []
#     if 'rating_inputs' not in st.session_state:
#         st.session_state['rating_inputs'] = []

#     def add_rating_input():
#         st.session_state['rating_inputs'].append({'movie': '', 'rating': 0.0})

#     def remove_rating_input(index):
#         st.session_state['rating_inputs'].pop(index)

#     if st.button('Thêm đánh giá'):
#         add_rating_input()

#     for index, rating_input in enumerate(st.session_state['rating_inputs']):
#         movie_col, rating_col, remove_col = st.columns([3, 1, 1])
#         rating_input['movie'] = movie_col.selectbox(
#             'Phim', list(movie_dict.keys()), key=f'movie_{index}', index=0)
#         rating_input['rating'] = rating_col.selectbox(
#             'Điểm', [1, 2, 3, 4, 5], key=f'rating_{index}')
#         if remove_col.button('Xoá', key=f'remove_{index}'):
#             remove_rating_input(index)

#     user_ratings = {movie_dict[input['movie']]: input['rating'] for input in st.session_state['rating_inputs']}
#     if st.button('Đề xuất'):
#         recommendations_user_input = user_input_ratings_recommendations(user_ratings)
#         display_recommendations(recommendations_user_input, f"Phim đề xuất cho bạn")
#         print(user_ratings)
