

# PREPROCESS DATA

import pandas as pd

# --- 1. Load core files ---
# Movies metadata
movies = pd.read_csv(
    "data/ml-100k/u.item",
    sep="|", encoding="latin-1",
    names=["movie_id","title","release_date","video_release_date","IMDb_URL",
           "unknown","Action","Adventure","Animation","Children","Comedy","Crime",
           "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
           "Romance","Sci-Fi","Thriller","War","Western"]
)

# Ratings
ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t", names=["user_id","item_id","rating","timestamp"]
)

# Users (optional enrichment)
users = pd.read_csv(
    "data/ml-100k/u.user",
    sep="|", names=["user_id","age","gender","occupation","zip"]
)

# --- 2. Collapse one-hot genre flags into a genre_list ---
genre_cols = ["Action","Adventure","Animation","Children","Comedy","Crime",
              "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
              "Mystery","Romance","Sci-Fi","Thriller","War","Western"]

def collapse_genres(row):
    return [g for g in genre_cols if row[g] == 1]

movies["genre_list"] = movies.apply(collapse_genres, axis=1)

# --- 3. Enrich with ratings ---
# Compute average rating and rating count per movie
ratings_summary = ratings.groupby("item_id").agg(
    avg_rating=("rating","mean"),
    rating_count=("rating","count")
).reset_index().rename(columns={"item_id":"movie_id"})

# Merge into movies
movies_enriched = movies.merge(ratings_summary, on="movie_id", how="left")

# --- 4. Select lean schema ---
movies_final = movies_enriched[[
    "movie_id",
    "title",
    "release_date",
    "genre_list",
    "avg_rating",
    "rating_count"
]]


# BUSINESS LOGIC

# --- Imports ---
# Core LangChain modules
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Embeddings + Vectorstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Display helpers for Jupyter
from IPython.display import display, Markdown


# --- 1. Convert DataFrame rows into Documents ---
# Each movie row becomes a text chunk for retrieval
movie_docs = []
for _, row in movies_final.iterrows():
    text = f"""
    Movie: {row['title']}
    Release Date: {row['release_date']}
    Genres: {', '.join(row['genre_list'])}
    Average Rating: {row['avg_rating']:.2f}
    Rating Count: {row['rating_count']}
    """
    movie_docs.append(Document(page_content=text.strip()))

# ✅ At this point, movie_docs is a list of LangChain Document objects,
# each containing one movie’s metadata in clean text form.


# --- 2. Embeddings ---
# Convert text chunks into dense vectors using HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 3. Vectorstore ---
# Store embeddings in FAISS for efficient similarity search
vectorstore = FAISS.from_documents(movie_docs, embeddings)

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
chain_movies = (
    {
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
)
