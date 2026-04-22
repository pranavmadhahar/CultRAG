
# PREPROCESS DATA

import pandas as pd
import ast

# -----------------------------
# 1. Load core files
# -----------------------------
# Features and Echonest have 3 metadata rows at the top → skip them
features = pd.read_csv("data/fma-small/features.csv", skiprows=[0,1,2], low_memory=False)

# Load raw echonest file without skipping
echonest_raw = pd.read_csv("data/fma-small/echonest.csv", low_memory=False)

# Row 1 contains the real column names → set them as header
echonest_raw.columns = echonest_raw.iloc[1]

# Drop the first two metadata rows (row 0 and row 1)
echonest = echonest_raw.drop([0,1])

# Rename first column to track_id
echonest.rename(columns={echonest.columns[0]: "track_id"}, inplace=True)

# Tracks has a multi-index header (two rows) → load with header=[0,1]
tracks = pd.read_csv("data/fma-small/tracks.csv", header=[0,1], low_memory=False)

# Flatten multi-index headers
tracks.columns = ['_'.join(col).strip() for col in tracks.columns.values]

# The first column is messy → force it to be 'track_id'
tracks.rename(columns={tracks.columns[0]: "track_id"}, inplace=True)

# Drop the first row (it only contains the string 'track_id' and NaNs)
tracks = tracks.drop(index=0)

# Genres is a simple lookup table (genre_id → title, parent)
genres = pd.read_csv("data/fma-small/genres.csv", low_memory=False)

# To avoid surprises, normalize all IDs right after loading
for df in [tracks, features, echonest]:
    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")

# Rename the messy first column to track_id
tracks.rename(columns={tracks.columns[0]: "track_id"}, inplace=True)

# -----------------------------
# 3. Parse multi-genre column
# -----------------------------
# track_genres_all is stored as a stringified list → convert to Python list
tracks["track_genres_all"] = tracks["track_genres_all"].apply(
    lambda x: ast.literal_eval(x) if pd.notnull(x) else []
)

# -----------------------------
# 4. Explode genres into rows
# -----------------------------
# Each track with multiple genres becomes multiple rows
tracks_exploded = tracks.explode("track_genres_all")

# -----------------------------
# 5. Merge with genres.csv
# -----------------------------
# Map genre IDs to human-readable titles
tracks_genres_named = tracks_exploded.merge(
    genres, left_on="track_genres_all", right_on="genre_id", how="left"
)

# -----------------------------
# 6. Group genres back into lists
# -----------------------------
# Aggregate all genre titles per track into a list
tracks_grouped = (
    tracks_genres_named.groupby("track_id")["title"]
    .apply(list)
    .reset_index()
    .rename(columns={"title": "genre_list"})
)

# -----------------------------
# 7. Merge genre list back into core tracks
# -----------------------------
tracks_named = tracks.merge(tracks_grouped, on="track_id", how="left")

# -----------------------------
# 8. Merge with features and echonest
# -----------------------------
songs_enriched = tracks_named.merge(features, left_on="track_id", right_on="track_id", how="left")
songs_enriched = songs_enriched.merge(echonest, on="track_id", how="left")

# --- Safeguard: coalesce artist_name_x and artist_name_y ---
songs_enriched["artist_name"] = songs_enriched["artist_name_x"].fillna(songs_enriched["artist_name_y"])

# If both are missing, fill with a placeholder
songs_enriched["artist_name"] = songs_enriched["artist_name"].fillna("Unknown Artist")

# Drop duplicates
songs_enriched.drop(columns=["artist_name_x", "artist_name_y"], inplace=True)

# --- Keep the required params ---
songs_final = songs_enriched[[
    "track_id",
    "track_title",
    "artist_name",
    "album_title",
    "genre_list"
]]

# -----------------------------
# ✅ Final DataFrame: songs_enriched
# -----------------------------
# Contains:
# - track_track_id (primary key)
# - album_title, artist_name, track_genre_top
# - genre_list (all human-readable genres)
# - 518 numeric audio features
# - Echonest descriptors (acousticness, danceability, energy, etc.)

# BUSINESS LOGIC

# --- Imports ---
# Core LangChain modules
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Embeddings + Vectorstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings


# Display helpers for Jupyter
from IPython.display import display, Markdown


# --- 1. Convert DataFrame rows into Documents ---
# Each movie row becomes a text chunk for retrieval
song_docs = []
for _, row in songs_final.iterrows():
    text = f"""
    Track: {row['track_title']}
    Artist: {row['artist_name']}
    Album: {row['album_title']}
    Genres: {', '.join(str(g) for g in row['genre_list'] if pd.notna(g))}
    """
    song_docs.append(Document(page_content=text.strip()))

# ✅ At this point, movie_docs is a list of LangChain Document objects,
# each containing one movie’s metadata in clean text form.


# --- 2. Embeddings ---
# Convert text chunks into dense vectors using HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 3. Vectorstore ---
# Store embeddings in FAISS for efficient similarity search
vectorstore = FAISS.from_documents(song_docs, embeddings)

# --- 4. Retriever ---
# Retriever is the interface to query FAISS
retriever = vectorstore.as_retriever()


# --- 5. LLM ---
# Define the language model (OpenAI in this case)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 6. Prompt Template ---
# Instructions for how the assistant should answer
prompt = ChatPromptTemplate.from_template("""
You are a retrieval‑augmented assistant (RAG). 
Use ONLY the provided context to answer the question. 
If the answer is not in the context, say "Not found in the catalog."

Context:
{context}

Question:
{question}

Rules:
- Do NOT add items that are not in the context.
- Do NOT guess or hallucinate.
- Output a valid markdown table with headers.
- Include a short summary after the table.

Answer:
""")


# --- 7. LCEL Retrieval Chain ---
# Helper to format retrieved docs into a single string
def format_docs(docs): 
    return "\n\n".join([d.page_content for d in docs])

# Build the chain: Retriever → Prompt → LLM
chain_songs = (
    {
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
)
