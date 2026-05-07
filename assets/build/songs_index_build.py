
"""
songs_index_build.py
--------------------
Build FAISS vector index from cleaned FMA-small songs dataset.

Purpose:
- Load cleaned song metadata (output of preprocessing module)
- Convert each song into a semantic text document
- Generate embeddings using HuggingFace sentence-transformers
- Store vectors in FAISS for fast similarity search (RAG-ready retrieval layer)

Input:
- ../cleaned_data/songs_clean.csv

Output:
- ../vectorstores/faiss_songs_index (FAISS vector store)
"""

import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


def build_songs_index():

    # =========================================================
    # STEP 1: LOAD CLEANED DATASET
    # =========================================================
    songs = pd.read_csv("../cleaned_data/songs_clean.csv")

    # =========================================================
    # STEP 1.1: DATA CLEANING + SAFETY FILTERS (CRITICAL)
    # =========================================================
    required_cols = ["track_title", "artist_name", "album_title"]

    # Remove missing core fields
    songs = songs.dropna(subset=required_cols)

    # Remove empty / whitespace / "nan" strings
    for col in required_cols:
        songs = songs[
            songs[col].astype(str).str.strip() != ""
        ]
        songs = songs[
            songs[col].astype(str).str.lower() != "nan"
        ]

    # Remove numeric-only junk titles ("2", "1", etc.)
    songs = songs[
        ~songs["track_title"].astype(str).str.fullmatch(r"\d+")
    ]

    # Remove very short corrupted titles
    songs = songs[
        songs["track_title"].astype(str).str.len() > 2
    ]

    # =========================================================
    # STEP 1.2: GENRE NORMALIZATION (IMPORTANT FIX)
    # =========================================================
    # Ensures embeddings always receive a consistent string format
    # Prevents list / NaN / mixed-type corruption

    songs["genre_list"] = songs["genre_list"].fillna("Unknown")

    songs["genre_list"] = songs["genre_list"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )

    # =========================================================
    # STEP 2: CONVERT EACH SONG → SEMANTIC DOCUMENT
    # =========================================================
    song_docs = []

    for _, row in songs.iterrows():

        # Structured semantic representation for embeddings
        # This improves similarity grouping (artist + genre + album alignment)

        text = f"""
        Title: {row['track_title']}. 
        Performed by {row['artist_name']}. 
        Released under the album {row['album_title']}. 
        Genres include: {row['genre_list']}. 
        This is a music track used for semantic search and recommendation in a songs catalog.
        """

        # =========================================================
        # STEP 2.1: CLEAN WHITESPACE NOISE
        # =========================================================
        text = " ".join(text.split())

        song_docs.append(
            Document(page_content=text)
        )

    print(f"✅ Prepared {len(song_docs)} song documents.")

    # =========================================================
    # STEP 3: INITIALIZE EMBEDDING MODEL
    # =========================================================
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # =========================================================
    # STEP 4: BUILD FAISS VECTOR INDEX
    # =========================================================
    vectorstore = FAISS.from_documents(song_docs, embeddings)

    # =========================================================
    # STEP 5: SAVE INDEX TO DISK
    # =========================================================
    output_path = "../vectorstores/faiss_songs_index"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    vectorstore.save_local(output_path)

    print(f"🚀 Songs FAISS index built and saved at {output_path}")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    build_songs_index()
