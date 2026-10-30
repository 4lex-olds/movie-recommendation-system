import streamlit as st
from recommender import MovieRecommender

st.set_page_config(page_title="Movie Match", page_icon="🎬", layout="wide")
st.title("🎬 Movie Match")
st.write("Find films with similar genres and descriptions using TF-IDF and cosine similarity.")
st.caption("Demo catalog only; recommendations are based on metadata similarity, not user ratings.")

@st.cache_resource
def load_recommender():
    return MovieRecommender()

recommender = load_recommender()
selected = st.selectbox("Choose a movie you like", recommender.titles())
count = st.slider("Number of recommendations", 3, 10, 5)

if st.button("Recommend movies", type="primary"):
    results = recommender.recommend(selected, n=count)
    for _, movie in results.iterrows():
        with st.container(border=True):
            st.subheader(movie["title"])
            st.caption(movie["genres"])
            st.write(movie["description"])
            st.progress(min(max(float(movie["similarity"]), 0.0), 1.0))
            st.caption(f"Metadata similarity: {movie['similarity']:.0%}")
