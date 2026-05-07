
"""
CultRAG.py
======================================================================
Multi-Domain RAG Orchestrator (Books + Movies + Songs)

🧠 ARCHITECTURE OVERVIEW
------------------------
This system is a 3-layer RAG pipeline:

1. INPUT LAYER
   - Normalizes memory + direct queries

2. ROUTING LAYER
   - Rewrites query for consistency (LLM-based router)
   - Routes to domain-specific RAG chains

3. DOMAIN RAG LAYER
   - BooksRAG / MoviesRAG / SongsRAG
   - Each returns structured JSON (no hallucination layer)

4. FALLBACK LAYER
   - Handles general queries outside catalog

5. PRESENTATION LAYER
   - Converts structured JSON → human-readable output

======================================================================

⚙️ DESIGN PRINCIPLES
--------------------
✔ Minimize hallucination (JSON-first design)
✔ Keep RAG outputs deterministic
✔ Keep LLM only where needed (routing + narration)
✔ Domain separation enforced at router level
✔ Memory-safe conversation handling
"""

# =========================================================
# STEP 1: ENVIRONMENT SETUP
# =========================================================
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

model = "gpt-4o-mini"

# =========================================================
# STEP 2: IMPORT DOMAIN CHAINS (RAG LAYERS)
# =========================================================
import sys, os
sys.path.append(os.path.abspath(".."))

from chain_books import chain_books
from chain_movies import chain_movies
from chain_songs import chain_songs

# =========================================================
# STEP 3: CORE LANGCHAIN IMPORTS
# =========================================================
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate
)
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# =========================================================
# STEP 4: LLM INITIALIZATION
# =========================================================
llm = ChatOpenAI(model=model, temperature=0.0)

# =========================================================
# STEP 5: INPUT NORMALIZATION LAYER
# =========================================================
def normalize_input(x):
    """
    WHY THIS EXISTS:
    ----------------
    LangChain memory returns:
    - list[BaseMessage] OR dict input

    This step ensures:
    ✔ consistent schema for downstream chains
    ✔ avoids KeyError in routing
    """

    if isinstance(x, list):
        return {
            "question": x[-1].content,
            "history": "\n".join(m.content for m in x[:-1])
        }

    return {
        "question": x.get("question", ""),
        "history": x.get("history", "")
    }

input_mapper = RunnableLambda(normalize_input)

# =========================================================
# STEP 6: ROUTER (LLM-BASED QUERY NORMALIZER)
# =========================================================
router_prompt = PromptTemplate(
    template="""
You are a query rewriting system.

Conversation history:
{history}

User question:
{question}

TASK:
- Rewrite ONLY if needed
- Keep meaning unchanged
- Return ONLY JSON

FORMAT:
{{
  "question": "final rewritten question"
}}
""",
    input_variables=["history", "question"]
)

router_chain = router_prompt | llm | JsonOutputParser()

# =========================================================
# STEP 7: DEFAULT FALLBACK CHAIN
# =========================================================
default_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

User question:
{question}

Return ONLY valid JSON:

{
  "domain": "general",
  "query_understanding": "what user wants",
  "answer": "helpful response"
}
""")

default_chain = (
    RunnableLambda(lambda x: {"question": x["question"]})
    | default_prompt
    | llm
    | JsonOutputParser()
)

# =========================================================
# STEP 8: RULE-BASED MULTI-DOMAIN ROUTER
# =========================================================
def multi_route(x):
    """
    WHY RULE-BASED ROUTER EXISTS:
    -----------------------------
    ✔ avoids extra LLM calls
    ✔ deterministic domain routing
    ✔ faster + cheaper than semantic routing
    """

    q = x["question"].lower()
    results = []

    if "book" in q:
        results.append(chain_books.invoke(x))

    if "movie" in q:
        results.append(chain_movies.invoke(x))

    if "song" in q or "music" in q:
        results.append(chain_songs.invoke(x))

    # fallback
    if not results:
        return default_chain.invoke(x)

    # unify multi-domain outputs
    return {
        "results": results
    }

router = RunnableLambda(multi_route)

# =========================================================
# STEP 9: CORE PIPELINE (RAG ORCHESTRATION)
# =========================================================
cult_chain_core = input_mapper | router_chain | router

# =========================================================
# STEP 10: MEMORY STORE (SESSION HANDLING)
# =========================================================
store = {}

def get_session_history(session_id: str):
    """
    Simple in-memory session store.
    Replace with Redis/Postgres in production.
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

cult_chain = RunnableWithMessageHistory(
    runnable=cult_chain_core,
    get_session_history=get_session_history,
    input_key="question",
    history_key="history"
)

# =========================================================
# STEP 11: NARRATION LAYER (HUMAN RESPONSE BUILDER)
# =========================================================
narrator_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Convert structured JSON into a clean readable response.

RULES:
- Be concise
- Use bullets or sections
- Do NOT hallucinate
- Use ONLY provided JSON

INPUT:
{data}
""")

narrator_chain = narrator_prompt | llm

def format_for_narrator(x):
    return {"data": x}

# =========================================================
# STEP 12: FINAL END-TO-END PIPELINE
# =========================================================
final_chain = (
    cult_chain
    | RunnableLambda(format_for_narrator)
    | narrator_chain
)
