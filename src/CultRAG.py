"""
CultRAG.py (Clean LCEL Orchestration - FIXED + STABLE)

Fixes:
- Proper input normalization (list → dict)
- PromptTemplate always receives mapping type
- Memory-safe LCEL pipeline
- Rule-based routing
- JSON router output
- Fully LCEL-native
"""

# --- Step 1: Environment ---
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

model = "gpt-4o-mini"

# --- Step 2: Domain Chains ---
import sys, os
sys.path.append(os.path.abspath(".."))

from chain_books import chain_books
from chain_movies import chain_movies
from chain_songs import chain_songs

# --- Step 3: Core Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# --- Step 4: LLM ---
llm = ChatOpenAI(model=model, temperature=0.0)

# =========================================================
# INPUT NORMALIZATION (CRITICAL)
# =========================================================
def normalize_input(x):
    """
    Handles BOTH:
    1. list[BaseMessage] from memory wrapper
    2. dict from direct invocation
    """

    # Case 1: memory gives message list
    if isinstance(x, list):
        return {
            "question": x[-1].content,
            "history": "\n".join(m.content for m in x[:-1])
        }

    # Case 2: normal dict input
    return {
        "question": x.get("question", ""),
        "history": x.get("history", "")
    }

input_mapper = RunnableLambda(normalize_input)

# --- Step 5: Router Prompt (JSON output) ---
router_prompt = PromptTemplate(
    template="""
You are a query rewriting system.

Conversation history:
{history}

User question:
{question}

Task:
- Rewrite question only if needed to make it self-contained
- If already clear, return unchanged
- Output ONLY valid JSON

Return format:
{{
  "question": "final rewritten question"
}}
""",
    input_variables=["history", "question"]
)

# --- Step 6: Parser ---
parser = JsonOutputParser()

router_chain = router_prompt | llm | parser

# --- Step 7: Default Chain ---
default_chain = (
    RunnableLambda(lambda x: x["question"])
    | llm
)

# --- Step 8: Rule-based Router ---
def multi_route(x):
    q = x["question"].lower()
    results = []

    if "book" in q:
        res = chain_books.invoke(x)
        results.append(res.content)

    if "movie" in q:
        res = chain_movies.invoke(x)
        results.append(res.content)


    if "song" in q or "music" in q:
        res = chain_songs.invoke(x)
        results.append(res.content)

    # fallback
    if not results:
        return default_chain.invoke(x).content

    return "\n\n".join(results)

router = RunnableLambda(multi_route)

# --- Step 9: CORE PIPELINE (FIXED ORDER) ---
cult_chain_core = input_mapper | router_chain | router

# --- Step 10: Memory Store ---
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# --- Step 11: FINAL CHAIN ---
cult_chain = RunnableWithMessageHistory(
    runnable=cult_chain_core,
    get_session_history=get_session_history,
    input_key="question",
    history_key="history"
)
