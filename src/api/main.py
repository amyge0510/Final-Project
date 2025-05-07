import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from ..retrieval.graph_rag import GraphRAG

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Amazon Knowledge Graph API",
    description="API for querying Amazon product data using GraphRAG",
    version="1.0.0"
)

# Initialize GraphRAG
graph_rag = GraphRAG()

class Query(BaseModel):
    text: str
    max_hops: int = 2

class Answer(BaseModel):
    question: str
    answer: str

@app.post("/query", response_model=Answer)
async def query_knowledge_graph(query: Query):
    """Query the knowledge graph using GraphRAG."""
    try:
        answer = graph_rag.answer_query(query.text)
        return Answer(question=query.text, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host=host, port=port) 