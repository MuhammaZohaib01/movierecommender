
import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies_cleaned.csv")
movies['combined'] = movies['overview'] + " " + movies['genres']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=4294056a772a8a1439a5363729b9d6bf&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
    return "https://via.placeholder.com/300x450.png?text=No+Image"

def recommend(movie_title):
    try:
        idx = movies[movies['title'] == movie_title].index[0]
    except:
        return [], [], [], [], []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    titles = movies['title'].iloc[movie_indices].values
    ids = movies['id'].iloc[movie_indices].values
    overviews = movies['overview'].iloc[movie_indices].values
    ratings = movies['vote_average'].iloc[movie_indices].values
    years = pd.to_datetime(movies['release_date'].iloc[movie_indices]).dt.year.values
    return titles, ids, overviews, ratings, years

st.set_page_config(page_title="Smart Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")
st.write("Select a movie from the dropdown to see similar ones with posters, overview, and ratings.")

selected_movie = st.selectbox("Choose a movie:", movies['title'].values)

if st.button("Get Recommendations"):
    titles, ids, overviews, ratings, years = recommend(selected_movie)
    cols = st.columns(5)
    for i in range(len(titles)):
        with cols[i]:
            st.image(fetch_poster(ids[i]), width=180)
            st.markdown("**" + titles[i] + "**")
            st.markdown("⭐ " + str(ratings[i]) + " | 📅 " + str(years[i]))
            st.caption(overviews[i][:150] + "...")
