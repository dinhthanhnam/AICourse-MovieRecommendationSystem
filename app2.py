import streamlit as st
import pandas as pd
from GUIHelper import hybrid, personalized_movies_recommendations, personalized_movies_recommendations_v2, id_map, ratings, movie_dict

def truncate_title_with_tooltip(title, max_length=15):
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
                    st.write("No image available.")

st.title('Movie Recommendation System')

# Add selectbox to choose between hybrid and personalized recommendations
recommendation_type = st.selectbox('Select recommendation type:', ['Hybrid Recommendations', 'Personalized Recommendations','Personalized Recommendations V2'])

# Display input options based on selected recommendation type
if recommendation_type == 'Hybrid Recommendations':
    selected_movie = st.selectbox('Select a movie:', list(movie_dict.keys()))
    user_id = st.number_input('Enter your user ID:', min_value=1, key="user_id_hybrid_input")
    if st.button('Get Recommendations'):
        movie_id = movie_dict[selected_movie]
        recommendations = hybrid(user_id, movie_id)
        display_recommendations(recommendations, "Recommended Movies")

elif recommendation_type == 'Personalized Recommendations':
    user_id_personalized = st.number_input('Enter your user ID:', min_value=1, key="user_id_personalized_input")
    if st.button('Get Personalized Recommendations'):
        recommendations_personalized = personalized_movies_recommendations(user_id_personalized)
        display_recommendations(recommendations_personalized, "Personalized Recommended Movies")

elif recommendation_type == 'Personalized Recommendations V2':
    user_id_personalized = st.number_input('Enter your user ID:', min_value=1, key="user_id_personalized_input")
    if st.button('Get Personalized Recommendations'):
        recommendations_personalized = personalized_movies_recommendations_v2(user_id_personalized)
        display_recommendations(recommendations_personalized, "Personalized Recommended Movies")
