
"""
books_index_build.py
--------------------
Load cleaned books dataset,
generate embeddings,
and build FAISS index.
"""

import pandas as pd

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


def build_books_index():

    # Step 1: Load cleaned CSV
    books = pd.read_csv("../cleaned_data/books_clean.csv")

    # Step 2: Convert rows into LangChain Documents
    book_docs = []

    for _, row in books.iterrows():

        text = f"""
        Title: {row['title']}. 
        Written by {row['authors']}. 
        Published in {row['original_publication_year']}. 
        It has an average rating of {row['rating']} and appears in {row['to_read_count']} reading lists. 
        Its genres include {row['tag_name']}.
        """

        text = " ".join(text.split())

        book_docs.append(Document(page_content=text))

    print(f"✅ Prepared {len(book_docs)} book documents.")

    # Step 3: Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 4: Create FAISS vector store
    vectorstore = FAISS.from_documents(
        book_docs,
        embeddings
    )

    # Step 5: Save FAISS index
    output_path = "../vectorstores/faiss_books_index"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vectorstore.save_local(output_path)



    print("✅ Books FAISS index built and saved.")


# Only run when executed directly
if __name__ == "__main__":
    build_books_index()
