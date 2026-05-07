
"""
movies_preprocess.py
---------------------
Preprocess MovieLens 100k dataset and build FAISS index.
"""

import pandas as pd


def build_movies_clean_csv():
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
        sep="\t",
        names=["user_id","item_id","rating","timestamp"]
    )

    # --- Step 2: Genres ---
    genre_cols = [
        "Action","Adventure","Animation","Children","Comedy","Crime",
        "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
        "Mystery","Romance","Sci-Fi","Thriller","War","Western"
    ]

    movies["genre_list"] = movies.apply(
        lambda row: ", ".join([g for g in genre_cols if row[g] == 1]),
        axis=1
    )

    # --- Step 3: Ratings summary ---
    ratings_summary = ratings.groupby("item_id").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index().rename(columns={"item_id": "movie_id"})

    movies_enriched = movies.merge(ratings_summary, on="movie_id", how="left")

    # --- Step 4: Final dataset ---
    movies_final = movies_enriched[
        ["movie_id", "title", "release_date", "genre_list", "avg_rating", "rating_count"]
    ]

    print("Processed movies:", len(movies_final))

    # --- Step 5: Save cleaned CSV ---
    movies_final.to_csv(
        "../cleaned_data/movies_clean.csv",
        index=False
    )

    print("✅ Cleaned movies dataset saved at ../cleaned_data/movies_clean.csv")

    return movies_final


if __name__ == "__main__":
    build_movies_clean_csv()
