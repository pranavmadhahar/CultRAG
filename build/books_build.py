"""
books_build.py
----------------
Preprocess the GoodBooks-10k dataset and build FAISS index.
This script is meant to be run once to generate embeddings and save the index.
"""

import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.utils.paths import DATA_DIR


def build_books_index():
    # --- PREPROCESS DATA ---
    books = pd.read_csv(DATA_DIR / "goodbooks-10k" / "books.csv")
    book_tags = pd.read_csv(DATA_DIR / "goodbooks-10k" / "book_tags.csv")
    tags = pd.read_csv(DATA_DIR / "goodbooks-10k" / "tags.csv")
    ratings_books = pd.read_csv(DATA_DIR / "goodbooks-10k" / "ratings.csv")
    to_read = pd.read_csv(DATA_DIR / "goodbooks-10k" / "to_read.csv")

    # Step 1: Join book_tags with tags → enrich with tag names
    book_tags_named = book_tags.merge(tags, on="tag_id")

    # Step 2: Merge with books using goodreads_book_id → attach metadata
    books_enriched = books.merge(book_tags_named, on="goodreads_book_id", how="left")

    # Step 3: Compute average rating per book_id → quality signal
    avg_ratings = ratings_books.groupby("book_id")["rating"].mean().reset_index()
    books_enriched = books_enriched.merge(avg_ratings, on="book_id", how="left")

    # Step 4: Count how many users marked each book as to-read → popularity signal
    to_read_count = to_read.groupby("book_id").size().reset_index(name="to_read_count")
    books_enriched = books_enriched.merge(to_read_count, on="book_id", how="left")

    # Step 5: Clean up NaN values for readability
    books_enriched["rating"] = books_enriched["rating"].fillna("No rating")
    books_enriched["tag_name"] = books_enriched["tag_name"].fillna("No tags")

    # Step 6: Group tags into a list per book
    books_grouped = books_enriched.groupby(
        ["book_id", "title", "authors", "original_publication_year", "rating", "to_read_count"]
    )["tag_name"].apply(list).reset_index()

    # Step 7: Convert DataFrame rows into LangChain Documents
    book_docs = []
    for _, row in books_grouped.iterrows():
        text = f"""
        Book: {row['title']}
        Authors: {row['authors']}
        Publication Year: {row['original_publication_year']}
        Rating: {row['rating']}
        Reading Count: {row['to_read_count']}
        Genres: {', '.join(row['tag_name'])}
        """
        book_docs.append(Document(page_content=text.strip()))

    print(f"✅ Prepared {len(book_docs)} book documents.")

    # Step 8: Embeddings → convert text chunks into dense vectors
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 9: Vectorstore → store embeddings in FAISS for efficient similarity search
    vectorstore = FAISS.from_documents(book_docs, embeddings)

    # Step 10: Save FAISS index
    vectorstore.save_local("../data/faiss_books_index")
    print("✅ Books FAISS index built and saved at ../data/faiss_books_index")


# Only run build when executed directly, not on import
if __name__ == "__main__":
    build_books_index()
