
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
    # We assume preprocessing has already:
    # - resolved genres
    # - normalized missing values
    # - flattened metadata into clean columns

    songs = pd.read_csv("../cleaned_data/songs_clean.csv")

    # =========================================================
    # STEP 2: CONVERT EACH SONG → NATURAL LANGUAGE DOCUMENT
    # =========================================================
    # Why?
    # Embedding models perform best on semantic text, not tabular data.
    # So we transform structured rows into descriptive "mini-documents".

    song_docs = []

    for _, row in songs.iterrows():

        # Create a human-readable semantic representation of a song
        # This helps embeddings capture relationships like:
        # artist ↔ genre ↔ album ↔ track identity

        text = f"""
        Title: {row['track_title']}. 
        Performed by {row['artist_name']}. 
        Released under the album {row['album_title']}. 
        Its genres include {row['genre_list']}.
        """

        # =========================================================
        # STEP 2.1: CLEAN WHITESPACE NOISE
        # =========================================================
        # Removes:
        # - newlines
        # - multiple spaces
        # - indentation artifacts from f-string formatting
        #
        # This ensures consistent tokenization for embeddings.

        text = " ".join(text.split())

        # Wrap into LangChain Document format
        song_docs.append(Document(page_content=text))

    print(f"✅ Prepared {len(song_docs)} song documents.")

    # =========================================================
    # STEP 3: INITIALIZE EMBEDDING MODEL
    # =========================================================
    # Using a lightweight, high-performance transformer model
    # suitable for semantic similarity tasks.

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # =========================================================
    # STEP 4: BUILD FAISS VECTOR INDEX
    # =========================================================
    # FAISS converts embeddings into a fast similarity search index.
    # This enables:
    # - nearest-neighbor search
    # - semantic retrieval
    # - foundation for RAG systems

    vectorstore = FAISS.from_documents(song_docs, embeddings)

    # =========================================================
    # STEP 5: SAVE INDEX TO DISK
    # =========================================================
    # Persist vector database so it can be reused without recomputation

    output_path = "../vectorstores/faiss_songs_index"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vectorstore.save_local(output_path)

    print(f"✅ Songs FAISS index built and saved at {output_path}")


# =========================================================
# ENTRY POINT
# =========================================================
# Allows script execution directly from terminal or notebook

if __name__ == "__main__":
    build_songs_index()
