
"""
songs_preprocess.py
-------------------
ETL pipeline for FMA-small dataset.

Goal:
- Normalize raw music metadata
- Resolve genres into human-readable format
- Build clean dataset for downstream RAG / search / analytics

Output:
- ../cleaned_data/songs_clean.csv
"""

import pandas as pd
import ast


def build_songs_clean_csv():

    # =========================================================
    # STEP 1: LOAD RAW DATASETS
    # =========================================================
    # Each file contains a different slice of the music graph:
    # - tracks: core metadata (title, artist, album)
    # - features: audio features (MFCC, spectral, etc.)
    # - echonest: additional metadata enrichment
    # - genres: mapping from genre_id → genre_name

    features = pd.read_csv(
        "../data/fma-small/features.csv",
        skiprows=[0, 1, 2],
        low_memory=False
    )

    echonest_raw = pd.read_csv(
        "../data/fma-small/echonest.csv",
        low_memory=False
    )

    tracks = pd.read_csv(
        "../data/fma-small/tracks.csv",
        header=[0, 1],
        low_memory=False
    )

    genres = pd.read_csv(
        "../data/fma-small/genres.csv",
        low_memory=False
    )

    # =========================================================
    # STEP 2: NORMALIZE ECHONEST STRUCTURE
    # =========================================================
    # Fix broken header structure and standardize track_id column

    echonest_raw.columns = echonest_raw.iloc[1]
    echonest = echonest_raw.drop([0, 1])
    echonest.rename(columns={echonest.columns[0]: "track_id"}, inplace=True)

    # =========================================================
    # STEP 3: FLATTEN MULTI-INDEX TRACK COLUMNS
    # =========================================================
    # Convert hierarchical columns into flat names:
    # ('track', 'title') → 'track_title'

    tracks.columns = ['_'.join(col).strip() for col in tracks.columns.values]
    tracks.rename(columns={tracks.columns[0]: "track_id"}, inplace=True)
    tracks = tracks.drop(index=0)

    # =========================================================
    # STEP 4: ENSURE CONSISTENT ID TYPES
    # =========================================================
    # Prevent merge issues caused by string/int mismatches

    for df in [tracks, features, echonest]:
        df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")

    # =========================================================
    # STEP 5: PARSE GENRE LISTS SAFELY
    # =========================================================
    # Convert stringified lists → Python lists

    tracks["track_genres_all"] = tracks["track_genres_all"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )

    # =========================================================
    # STEP 6: EXPLODE GENRES (NORMALIZATION STEP)
    # =========================================================
    # Convert:
    # [genre1, genre2] → row per genre

    tracks_exploded = tracks.explode("track_genres_all")

    # =========================================================
    # STEP 7: MAP GENRE IDs → HUMAN READABLE NAMES
    # =========================================================

    tracks_genres_named = tracks_exploded.merge(
        genres,
        left_on="track_genres_all",
        right_on="genre_id",
        how="left"
    )

    # =========================================================
    # STEP 8: RE-GROUP GENRES PER TRACK
    # =========================================================
    # Reconstruct list of genre names per track_id

    tracks_grouped = (
        tracks_genres_named.groupby("track_id")["title"]
        .apply(list)
        .reset_index()
        .rename(columns={"title": "genre_list"})
    )

    # =========================================================
    # STEP 9: MERGE BACK INTO CORE TRACK DATA
    # =========================================================

    tracks_named = tracks.merge(tracks_grouped, on="track_id", how="left")

    # =========================================================
    # STEP 10: MERGE FEATURE + ENRICHMENT DATA
    # =========================================================

    songs_enriched = tracks_named.merge(features, on="track_id", how="left")
    songs_enriched = songs_enriched.merge(echonest, on="track_id", how="left")

    # =========================================================
    # STEP 11: RESOLVE ARTIST NAME CLEANLY
    # =========================================================
    # Prefer echonest value, fallback to tracks dataset

    songs_enriched["artist_name"] = songs_enriched["artist_name_x"].fillna(
        songs_enriched["artist_name_y"]
    )
    songs_enriched["artist_name"] = songs_enriched["artist_name"].fillna("Unknown Artist")

    songs_enriched.drop(columns=["artist_name_x", "artist_name_y"], inplace=True)

    # =========================================================
    # STEP 12: SELECT FINAL CLEAN SCHEMA
    # =========================================================
    # Keep only fields needed for downstream usage

    songs_final = songs_enriched[
        ["track_id", "track_title", "artist_name", "album_title", "genre_list"]
    ].copy()

    # =========================================================
    # STEP 13: CLEAN + SERIALIZE GENRE LIST
    # =========================================================
    # Convert:
    # list → "genre1, genre2"
    # Ensure no NaN contamination

    def safe_join_genres(x):
        if isinstance(x, list):
            return ", ".join(str(g) for g in x if pd.notna(g))
        return ""

    songs_final["genre_list"] = songs_final["genre_list"].apply(safe_join_genres)

    # =========================================================
    # STEP 14: SAVE CLEAN DATASET
    # =========================================================

    output_path = "../cleaned_data/songs_clean.csv"
    songs_final.to_csv(output_path, index=False)

    print(f"✅ Cleaned songs dataset saved at {output_path}")
    print(f"🎧 Total songs processed: {len(songs_final)}")

    return songs_final


if __name__ == "__main__":
    build_songs_clean_csv()
