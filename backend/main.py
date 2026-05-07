from fastapi import FastAPI
from pydantic import BaseModel
from src.CultRAG import final_chain, cult_chain

app = FastAPI()

class Query(BaseModel):
    question: str
    session_id: str = "default"

@app.post("/chat")
def text_response(query: Query):
    response = final_chain.invoke(
        query.question,
        config={"configurable": {"session_id": query.session_id}}
    )
    return response.content

@app.post("/structured")
def structured_response(query: Query):
    response = cult_chain.invoke(
        query.question,
        config={"configurable": {"session_id": query.session_id}}
    )
    return response