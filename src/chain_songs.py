"""
chain_songs.py
----------------
Core LangChain module for SongsRAG.

Responsibilities:
- Preprocess the FMA-small dataset (tracks, features, echonest, genres)
- Build embeddings and FAISS vectorstore
- Define a retrieval-augmented generation (RAG) chain for song queries
- Designed for integration into the CultRAG pipeline
"""

# --- PREPROCESS DATA ---
import pandas as pd
import ast

# Step 1: Load raw files
features = pd.read_csv("../data/fma-small/features.csv", skiprows=[0,1,2], low_memory=False)
echonest_raw = pd.read_csv("../data/fma-small/echonest.csv", low_memory=False)
tracks = pd.read_csv("../data/fma-small/tracks.csv", header=[0,1], low_memory=False)
genres = pd.read_csv("../data/fma-small/genres.csv", low_memory=False)

# Step 2: Normalize echonest headers
echonest_raw.columns = echonest_raw.iloc[1]
echonest = echonest_raw.drop([0,1])
echonest.rename(columns={echonest.columns[0]: "track_id"}, inplace=True)

# Step 3: Flatten tracks multi-index headers
tracks.columns = ['_'.join(col).strip() for col in tracks.columns.values]
tracks.rename(columns={tracks.columns[0]: "track_id"}, inplace=True)
tracks = tracks.drop(index=0)

# Step 4: Normalize IDs
for df in [tracks, features, echonest]:
    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")

# Step 5: Parse multi-genre column
tracks["track_genres_all"] = tracks["track_genres_all"].apply(
    lambda x: ast.literal_eval(x) if pd.notnull(x) else []
)

# Step 6: Explode genres into rows
tracks_exploded = tracks.explode("track_genres_all")

# Step 7: Merge with genres.csv → human-readable titles
tracks_genres_named = tracks_exploded.merge(
    genres, left_on="track_genres_all", right_on="genre_id", how="left"
)

# Step 8: Group genres back into lists
tracks_grouped = (
    tracks_genres_named.groupby("track_id")["title"]
    .apply(list)
    .reset_index()
    .rename(columns={"title": "genre_list"})
)

# Step 9: Merge genre list back into core tracks
tracks_named = tracks.merge(tracks_grouped, on="track_id", how="left")

# Step 10: Merge with features and echonest
songs_enriched = tracks_named.merge(features, on="track_id", how="left")
songs_enriched = songs_enriched.merge(echonest, on="track_id", how="left")

# Step 11: Safeguard artist_name
songs_enriched["artist_name"] = songs_enriched["artist_name_x"].fillna(songs_enriched["artist_name_y"])
songs_enriched["artist_name"] = songs_enriched["artist_name"].fillna("Unknown Artist")
songs_enriched.drop(columns=["artist_name_x", "artist_name_y"], inplace=True)

# Step 12: Select lean schema
songs_final = songs_enriched[[
    "track_id","track_title","artist_name","album_title","genre_list"
]]


# --- BUSINESS LOGIC ---

# Core LangChain modules
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Embeddings + Vectorstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 13: Convert DataFrame rows into LangChain Documents
song_docs = []
for _, row in songs_final.iterrows():
    text = f"""
    Track: {row['track_title']}
    Artist: {row['artist_name']}
    Album: {row['album_title']}
    Genres: {', '.join(str(g) for g in row['genre_list'] if pd.notna(g))}
    """
    song_docs.append(Document(page_content=text.strip()))

# ✅ At this point, song_docs is a list of LangChain Document objects,
# each containing one track’s metadata in clean text form.


# Step 14: Embeddings → convert text chunks into dense vectors
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 15: Vectorstore → store embeddings in FAISS
vectorstore = FAISS.from_documents(song_docs, embeddings)

# Step 16: Retriever → interface to query FAISS
retriever = vectorstore.as_retriever()

# Step 17: LLM → define the language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step 18: Prompt Template → instructions for how the assistant should answer
prompt = ChatPromptTemplate.from_template("""
You are a retrieval‑augmented assistant (RAG). 
Use ONLY the provided context to answer the question. 
If the answer is not in the context, say "Not found in the catalog."

Conversation so far:
{history}

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


# Step 19: Helper to format retrieved docs into a single string
def format_docs(docs):
    """
    Convert a list of LangChain Document objects into a single string.
    Each document’s page_content is joined with double newlines.
    Used to feed retrieved context into the prompt.
    """
    return "\n\n".join([d.page_content for d in docs])


# Step 20: Build the Core LCEL Retrieval Chain
chain_songs = (
    {
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"],
        "history": lambda x: x.get("history", "")
    }
    | prompt
    | llm
)
