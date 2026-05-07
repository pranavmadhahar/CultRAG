"""
chain_books.py
----------------
Core LangChain module for BooksRAG.

Responsibilities:
- Load the persisted FAISS index built by books_build.py
- Define a retrieval-augmented generation (RAG) chain for book queries
- Designed for integration into the CultRAG pipeline
"""

# --- BUSINESS LOGIC ---

# Core LangChain modules
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Embeddings + Vectorstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from utils.paths import VECTORSTORES_DIR


# Step 1: Load persisted FAISS index (built once in build/books_build.py)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    str(VECTORSTORES_DIR / "faiss_books_index"),
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

# Step 2: LLM → define the language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step 3: Prompt Template → instructions for how the assistant should answer

prompt_books = ChatPromptTemplate.from_template("""
You are a book recommendation assistant using retrieval-augmented generation (RAG).

You are given:
- Conversation history
- Retrieved book context
- A user question

Your task:
Extract ONLY relevant book information from the context and return structured results.

IMPORTANT RULES:
Rules:
- Always set domain = "books"
- ONLY include BOOKS in your answer.
- DO NOT include songs, movies, or any other media, even if present in the context.
- Ignore any non-books entries in the context completely.
- Do NOT add items that are not in the context.
- Do NOT guess or hallucinate.
- Do NOT mention other domains (songs, movies) or their absence.
- Include a short summary after the table.

Conversation History:
{history}

Context:
{context}

Question:
{question}

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "domain": "movies",
  "query_understanding": "brief interpretation of what user wants",
  "results": [
    {{
      "title": "...",
      "writer": "...",
      "publication_year": "...",
      "avg_rating": "...",
      "reading_count": "...",
      "genres": "..."
    }}
  ],
  "summary": "short explanation of the results in 1-2 lines"
}}

FINAL RULE:
Return ONLY valid JSON. No markdown, no tables, no extra text, no extra formatting.
""")

# Step 4: Helper to format retrieved docs into a single string
def format_docs(docs):
    """
    Convert a list of LangChain Document objects into a single string.
    Each document’s page_content is joined with double newlines.
    Used to feed retrieved context into the prompt.
    """
    return "\n\n".join([d.page_content for d in docs])

parser = JsonOutputParser()

# Step 5: Build the Core LCEL Retrieval Chain
chain_books = (
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
        "history": lambda x: x.get("history", "")
    }
    | prompt_books
    | llm
    | parser
)
