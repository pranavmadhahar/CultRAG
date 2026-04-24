"""
movies_build.py
----------------
Preprocess the MovieLens 100k dataset and build FAISS index.
Run this script once to generate embeddings and save the index.
"""

import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_movies_index():
    # --- Step 1: Load raw MovieLens files ---
    movies = pd.read_csv(
        "../data/ml-100k/u.item",
        sep="|", encoding="latin-1",
        names=["movie_id","title","release_date","video_release_date","IMDb_URL",
               "unknown","Action","Adventure","Animation","Children","Comedy","Crime",
               "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
               "Romance","Sci-Fi","Thriller","War","Western"]
    )

    ratings = pd.read_csv(
        "../data/ml-100k/u.data",
        sep="\t", names=["user_id","item_id","rating","timestamp"]
    )

    users = pd.read_csv(
        "../data/ml-100k/u.user",
        sep="|", names=["user_id","age","gender","occupation","zip"]
    )

    # --- Step 2: Collapse one-hot genre flags into a genre_list ---
    genre_cols = ["Action","Adventure","Animation","Children","Comedy","Crime",
                  "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
                  "Mystery","Romance","Sci-Fi","Thriller","War","Western"]

    def collapse_genres(row):
        return [g for g in genre_cols if row[g] == 1]

    movies["genre_list"] = movies.apply(collapse_genres, axis=1)

    # --- Step 3: Enrich with ratings (average + count) ---
    ratings_summary = ratings.groupby("item_id").agg(
        avg_rating=("rating","mean"),
        rating_count=("rating","count")
    ).reset_index().rename(columns={"item_id":"movie_id"})

    movies_enriched = movies.merge(ratings_summary, on="movie_id", how="left")

    # --- Step 4: Select lean schema ---
    movies_final = movies_enriched[[
        "movie_id","title","release_date","genre_list","avg_rating","rating_count"
    ]]

    # --- Step 5: Convert rows into LangChain Documents ---
    movie_docs = []
    for _, row in movies_final.iterrows():
        text = f"""
        Movie: {row['title']}
        Release Year: {row['release_date']}
        Genres: {', '.join(row['genre_list'])}
        Average Rating: {row['avg_rating']}
        Rating Count: {row['rating_count']}
        """
        movie_docs.append(Document(page_content=text.strip()))

    print(f"✅ Prepared {len(movie_docs)} movie documents.")

    # --- Step 6: Build embeddings + FAISS index ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(movie_docs, embeddings)

    # --- Step 7: Save index ---
    vectorstore.save_local("../data/faiss_movies_index")
    print("✅ Movies FAISS index built and saved at ../data/faiss_movies_index")

if __name__ == "__main__":
    build_movies_index()
