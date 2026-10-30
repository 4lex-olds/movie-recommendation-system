from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "movies.csv"

class MovieRecommender:
    def __init__(self, data_path=DATA_PATH):
        self.movies = pd.read_csv(data_path).fillna("")
        required = {"title", "genres", "description"}
        missing = required - set(self.movies.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        self.movies["features"] = (
            self.movies["genres"].astype(str) + " " +
            self.movies["description"].astype(str)
        )
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.movies["features"])
        self.similarity = cosine_similarity(self.matrix)

    def titles(self):
        return self.movies["title"].tolist()

    def recommend(self, title, n=5):
        matches = self.movies.index[self.movies["title"].str.casefold() == title.casefold()].tolist()
        if not matches:
            raise ValueError(f"Movie not found: {title}")
        idx = matches[0]
        ranked = sorted(
            enumerate(self.similarity[idx]),
            key=lambda item: item[1],
            reverse=True
        )
        ranked = [(i, score) for i, score in ranked if i != idx][:n]
        result = self.movies.iloc[[i for i, _ in ranked]][["title", "genres", "description"]].copy()
        result["similarity"] = [float(score) for _, score in ranked]
        return result.reset_index(drop=True)
