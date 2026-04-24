"""
songs_build.py
---------------
Preprocess the FMA-small dataset and build FAISS index.
Run this script once to generate embeddings and save the index.
"""

import pandas as pd
import ast
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ✅ centralized path system
from src.utils.paths import DATA_DIR


def build_songs_index():
    # --- Step 1: Load raw files ---
    features = pd.read_csv(
        DATA_DIR / "fma-small" / "features.csv",
        skiprows=[0, 1, 2],
        low_memory=False
    )

    echonest_raw = pd.read_csv(
        DATA_DIR / "fma-small" / "echonest.csv",
        low_memory=False
    )

    tracks = pd.read_csv(
        DATA_DIR / "fma-small" / "tracks.csv",
        header=[0, 1],
        low_memory=False
    )

    genres = pd.read_csv(
        DATA_DIR / "fma-small" / "genres.csv",
        low_memory=False
    )

    # --- Step 2: Normalize echonest headers ---
    echonest_raw.columns = echonest_raw.iloc[1]
    echonest = echonest_raw.drop([0, 1])
    echonest.rename(columns={echonest.columns[0]: "track_id"}, inplace=True)

    # --- Step 3: Flatten multi-index headers ---
    tracks.columns = ['_'.join(col).strip() for col in tracks.columns.values]
    tracks.rename(columns={tracks.columns[0]: "track_id"}, inplace=True)
    tracks = tracks.drop(index=0)

    # --- Step 4: Normalize IDs ---
    for df in [tracks, features, echonest]:
        df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")

    # --- Step 5: Parse multi-genre column ---
    tracks["track_genres_all"] = tracks["track_genres_all"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )

    # --- Step 6: Explode genres ---
    tracks_exploded = tracks.explode("track_genres_all")

    # --- Step 7: Merge with genres.csv ---
    tracks_genres_named = tracks_exploded.merge(
        genres,
        left_on="track_genres_all",
        right_on="genre_id",
        how="left"
    )

    # --- Step 8: Group genres back ---
    tracks_grouped = (
        tracks_genres_named.groupby("track_id")["title"]
        .apply(list)
        .reset_index()
        .rename(columns={"title": "genre_list"})
    )

    # --- Step 9: Merge back into tracks ---
    tracks_named = tracks.merge(tracks_grouped, on="track_id", how="left")

    # --- Step 10: Merge features + echonest ---
    songs_enriched = tracks_named.merge(features, on="track_id", how="left")
    songs_enriched = songs_enriched.merge(echonest, on="track_id", how="left")

    # --- Step 11: Artist name cleanup ---
    songs_enriched["artist_name"] = songs_enriched["artist_name_x"].fillna(
        songs_enriched["artist_name_y"]
    )
    songs_enriched["artist_name"] = songs_enriched["artist_name"].fillna("Unknown Artist")
    songs_enriched.drop(columns=["artist_name_x", "artist_name_y"], inplace=True)

    # --- Step 12: Lean schema ---
    songs_final = songs_enriched[
        ["track_id", "track_title", "artist_name", "album_title", "genre_list"]
    ]

    # --- Step 13: Convert to Documents ---
    song_docs = []
    for _, row in songs_final.iterrows():
        text = f"""
        Track: {row['track_title']}
        Artist: {row['artist_name']}
        Album: {row['album_title']}
        Genres: {', '.join(row['genre_list']) if isinstance(row['genre_list'], list) else ''}
        """
        song_docs.append(Document(page_content=text.strip()))

    print(f"✅ Prepared {len(song_docs)} song documents.")

    # --- Step 14: Embeddings ---
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Step 15: FAISS index ---
    vectorstore = FAISS.from_documents(song_docs, embeddings)

    # --- Step 16: Save index (FIXED) ---
    save_path = DATA_DIR / "faiss_songs_index"
    vectorstore.save_local(str(save_path))

    print(f"✅ Songs FAISS index built and saved at {save_path}")


if __name__ == "__main__":
    build_songs_index()