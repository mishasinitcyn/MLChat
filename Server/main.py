from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import CORS_ORIGINS
from typing import List
from utils import generate_response, generate_prompt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"], 
)

class QueryInput(BaseModel):
    query: str
    history: List[str] = []

@app.post("/query/")
async def handle_query(query_input: QueryInput):    
    try:
        prompt = generate_prompt(query_input.query, query_input.history)
        response_text = generate_response(prompt)
        print("Prompt: ", prompt)
        print("Response: ", response_text)
        return {"response": response_text}
    except Exception as e:
        print("Error: ", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    print("Server is starting...")

@app.on_event("shutdown")
async def shutdown_event():
    print("Server is shutting down...")