"""
chain_songs.py
----------------
Core LangChain module for SongsRAG.

Responsibilities:
- Load the persisted FAISS index (built once in build/songs_build.py)
- Define a retrieval-augmented generation (RAG) chain for song queries
- Designed for integration into the CultRAG pipeline
"""

# --- BUSINESS LOGIC ---

# Core LangChain modules
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Embeddings + Vectorstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils.paths import VECTORSTORES_DIR

# Step 1: Load persisted FAISS index (built once in build/songs_build.py)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    str(VECTORSTORES_DIR / "faiss_songs_index"),
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

# Step 2: LLM → define the language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step 3: Prompt Template → instructions for how the assistant should answer
prompt_songs = ChatPromptTemplate.from_template("""
You are a song recommendation assistant using retrieval-augmented generation (RAG).

Use ONLY the provided context to answer the question.
If the answer is not in the context, say "Not found in the catalog."

Conversation so far:
{history}

Context:
{context}

Question:
{question}

Rules:
- ONLY include SONGS in your answer.
- DO NOT include books, movies, or any other media, even if present in the context.
- Ignore any non-song entries in the context completely.
- Do NOT add items that are not in the context.
- Do NOT guess or hallucinate.
- Do NOT mention other domains (books, movies) or their absence.
- If context contains mixed types, extract and return ONLY song-related entries.
- Output a valid markdown table with headers.
- Include a short summary after the table.

Answer:
""")

# Step 4: Helper to format retrieved docs into a single string
def format_docs(docs):
    """
    Convert a list of LangChain Document objects into a single string.
    Each document’s page_content is joined with double newlines.
    Used to feed retrieved context into the prompt.
    """
    return "\n\n".join([d.page_content for d in docs])

# Step 5: Build the Core LCEL Retrieval Chain
chain_songs = (
    {
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"],
        "history": lambda x: x.get("history", "")
    }
    | prompt_songs
    | llm
)
