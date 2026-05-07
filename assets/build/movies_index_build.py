
"""
movies_index_build.py
--------------------
Load cleaned movies dataset,
generate embeddings,
and build FAISS index.
"""

import pandas as pd

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def build_movies_index():

    movies = pd.read_csv("../cleaned_data/movies_clean.csv")
    # --- Step 5: Convert rows into LangChain Documents ---
    movie_docs = []

    for _, row in movies.iterrows():
        text = f"""
        Title: {row['title']}. 
        Released in {row['release_date']}. 
        Genres include {row['genre_list']}. 
        It has an average rating of {row['avg_rating']} based on {row['rating_count']} user ratings.
        """

        text = " ".join(text.split())

        movie_docs.append(Document(page_content=text))

    print(f"✅ Prepared {len(movie_docs)} movie documents.")

    # --- Step 6: Build embeddings + FAISS index ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(movie_docs, embeddings)

    # --- Step 7: Save index ---
    output_path = "../vectorstores/faiss_movies_index"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vectorstore.save_local(output_path)


    print("✅ Movies FAISS index built and saved at ../data/faiss_movies_index")

if __name__ == "__main__":
    build_movies_index()
