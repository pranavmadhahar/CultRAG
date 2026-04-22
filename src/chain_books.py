
# PREPROCESS DATA
import pandas as pd

# Load core files
books = pd.read_csv("data/goodbooks-10k/books.csv")
book_tags = pd.read_csv("data/goodbooks-10k/book_tags.csv")
tags = pd.read_csv("data/goodbooks-10k/tags.csv")
ratings_books = pd.read_csv("data/goodbooks-10k/ratings.csv")
to_read = pd.read_csv("data/goodbooks-10k/to_read.csv")

# Step 1: Join book_tags with tags to get tag names
book_tags_named = book_tags.merge(tags, on="tag_id")

# Step 2: Merge with books using goodreads_book_id
books_enriched = books.merge(book_tags_named, on="goodreads_book_id", how="left")

# Step 3: Compute average rating per book_id
avg_ratings = ratings_books.groupby("book_id")["rating"].mean().reset_index()

# Step 4: Merge ratings into enriched books
books_enriched = books_enriched.merge(avg_ratings, on="book_id", how="left")

# Step 5: Compute how many user marked each book as to-read
to_read_count = to_read.groupby("book_id").size().reset_index(name="to_read_count")

# Step 6: Merge read_count into enriched books
books_enriched = books_enriched.merge(to_read_count, on="book_id", how="left")

# Optional: clean up NaN values
books_enriched["rating"] = books_enriched["rating"].fillna("No rating")
books_enriched["tag_name"] = books_enriched["tag_name"].fillna("No tags")

# Group tags into a list per book
books_grouped = books_enriched.groupby(
    ["book_id", "title", "authors", "original_publication_year", "rating", "to_read_count"]
)["tag_name"].apply(list).reset_index()


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
book_docs = []
for _, row in books_grouped.iterrows():
    text = f"""
    Book: {row['title']}
    Authors: {row['authors']}
    Publication Year: {row['original_publication_year']}
    Rating: {row['rating']:.2f}
    Reading Count: {row['to_read_count']}
    Genres: {', '.join(row['tag_name'])}
    """
    book_docs.append(Document(page_content=text.strip()))

# ✅ At this point, movie_docs is a list of LangChain Document objects,
# each containing one movie’s metadata in clean text form.


# --- 2. Embeddings ---
# Convert text chunks into dense vectors using HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 3. Vectorstore ---
# Store embeddings in FAISS for efficient similarity search
vectorstore = FAISS.from_documents(book_docs, embeddings)

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
chain_books = (
    {
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
)
