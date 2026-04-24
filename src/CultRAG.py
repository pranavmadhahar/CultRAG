"""
CultRAG.py (Core Orchestration)
-------------------------------
This module orchestrates the CultRAG pipeline:
- Loads domain chains (Books, Movies, Songs) from ../src/
- Defines an LLM-based router_chain that rewrites queries with context
- Dispatches to the correct domain chain using RunnableBranch
- Wraps cult_chain_core with RunnableWithMessageHistory for multi‑turn context
- Adds summarization of history when it grows too long
- Uses modern LCEL (LangChain Expression Language) operators for clarity and maintainability
"""

# --- Step 1: Environment Setup ---
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())   # Load API keys from .env

model = "gpt-4o-mini"

# --- Step 2: Import Domain Chains ---
import sys, os
sys.path.append(os.path.abspath(".."))  # go up one level from notebooks/

from src.chain_books import chain_books
from src.chain_movies import chain_movies
from src.chain_songs import chain_songs

# --- Step 3: Core LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# --- Step 4: Define Base LLM ---
llm = ChatOpenAI(model=model, temperature=0.0)

# --- Step 5: Default Fallback Chain ---
default_prompt = PromptTemplate(
    template="{question}",
    input_variables=["question"]
)
default_chain = default_prompt | llm

# --- Step 6: LLM Router Prompt ---
router_prompt = PromptTemplate(
    template="""
You are a router. Based on the conversation history and the latest user input,
decide whether the query needs rewriting.

Conversation so far:
{history}

User input:
{question}

Instructions:
- If the input is already clear and self-contained, keep it unchanged.
- If the input is vague or depends on prior context, rewrite it into a self-contained query.
- Always output both keys: "question" (rewritten or unchanged) and "history" (unchanged).

Output a JSON object with:
- question: the final query (rewritten or unchanged)
- history: the conversation history (unchanged)
""",
    input_variables=["history", "question"]
)

parser = JsonOutputParser()
router_chain = router_prompt | llm | parser

# --- Step 7: Router Dispatch ---
# --- Step 7: Router Dispatch with Debug ---
# --- Step 7: Router Dispatch with Debug ---

# --- Step 7: Router Dispatch with Debug ---
# --- Step 7: Router Dispatch with Debug ---
router = RunnableBranch(
    (
        lambda x: "book" in x["question"].lower(),
        RunnableLambda(lambda y: (print("Dispatch → Books") or chain_books.invoke(y)))
    ),
    (
        lambda x: "movie" in x["question"].lower(),
        RunnableLambda(lambda y: (print("Dispatch → Movies") or chain_movies.invoke(y)))
    ),
    (
        lambda x: "song" in x["question"].lower() or "music" in x["question"].lower(),
        RunnableLambda(lambda y: (print("Dispatch → Songs") or chain_songs.invoke(y)))
    ),
    # Default branch: just a runnable, not a tuple
    RunnableLambda(lambda y: (print("Dispatch → Default") or default_chain.invoke(y)))
)



# --- Step 8: CultRAG Core Pipeline ---
cult_chain_core = RunnableSequence(
    {"question": lambda x: x["input"], "history": lambda x: x["history"]},
    router_chain,
    RunnableLambda(lambda x: (print("Router output:", x), x)[1]),  # debug
    router
)

# --- Step 9: Summarizer Chain ---
summary_prompt = PromptTemplate(
    template="Summarize the following conversation briefly:\n\n{messages}",
    input_variables=["messages"]
)
summarizer_chain = summary_prompt | llm

def summarize_history(messages, max_len=5):
    # Keep only human messages (user inputs)
    user_msgs = [m.content for m in messages if m.type == "human"]

    if len(user_msgs) > max_len:
        summary = summarizer_chain.invoke({"messages": "\n".join(user_msgs)})
        return summary.content

    return "\n".join(user_msgs)


# --- Step 10: Wrap cult_chain_core with RunnableWithMessageHistory ---
shared_history = InMemoryChatMessageHistory()

cult_chain = RunnableWithMessageHistory(
    runnable=RunnableLambda(
        lambda inputs: cult_chain_core.invoke({
            "input": inputs[-1].content,  # last message content
            "history": summarize_history(shared_history.messages)
        })
    ),
    get_session_history=lambda _: shared_history,
    input_key="input",
    history_key="history"
)

