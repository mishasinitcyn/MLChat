from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import CORS_ORIGINS

from config import get_embeddings
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

@app.post("/query/")
async def handle_query(query_input: QueryInput):
    try:
        prompt = generate_prompt(query_input.query)
        print("Prompt generated: ", prompt)
        response_text = generate_response(prompt)
        print("Response generated: ", response_text)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    print("Server is starting... setup tasks here if any.")

@app.on_event("shutdown")
async def shutdown_event():
    print("Server is shutting down... cleanup tasks here if any.")