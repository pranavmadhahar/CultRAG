"""
books_build.py
----------------
Preprocess the GoodBooks-10k dataset and build FAISS index.
This script is meant to be run once to generate cleaned data.
"""

import pandas as pd


def build_books_clean_csv():
    # --- PREPROCESS DATA ---
    books = pd.read_csv("../data/goodbooks-10k/books.csv")
    book_tags = pd.read_csv("../data/goodbooks-10k/book_tags.csv")
    tags = pd.read_csv("../data/goodbooks-10k/tags.csv")
    ratings_books = pd.read_csv("../data/goodbooks-10k/ratings.csv")
    to_read = pd.read_csv("../data/goodbooks-10k/to_read.csv")

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

        # Step 7: Convert tag list → readable string
    books_grouped["tag_name"] = books_grouped["tag_name"].apply(
        lambda x: ", ".join(x)
    )

    # Step 8: Keep only useful columns
    books_grouped = books_grouped[
        [
            "title",
            "authors",
            "original_publication_year",
            "rating",
            "to_read_count",
            "tag_name"
        ]
    ]

    # Step 9: Save cleaned CSV
    books_grouped.to_csv(
        "../cleaned_data/books_clean.csv",
        index=False
    )

    print("✅ Cleaned books dataset saved at ../cleaned_data/books_clean.csv")



# Only run build when executed directly, not on import
if __name__ == "__main__":
    build_books_clean_csv()
