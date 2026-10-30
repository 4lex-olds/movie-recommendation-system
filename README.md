# Movie Recommendation System

A content-based recommender that suggests films using TF-IDF features built from genres, keywords, and plot-style descriptions.

## What it demonstrates
- Text preprocessing and feature engineering
- TF-IDF vectorization
- Cosine similarity for ranking recommendations
- A simple interactive Streamlit interface
- Explainable, reproducible recommendations from movie metadata

## Quick start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The included catalog is a small curated demo dataset so the app works immediately. Expand it with a properly licensed public dataset before presenting it as a broad recommendation engine.

## Project structure
- `data/movies.csv` — small movie catalog
- `recommender.py` — recommendation logic
- `app.py` — interactive interface
