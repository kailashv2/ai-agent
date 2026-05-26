from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent import run_agent
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

app = FastAPI(title="AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str
    session_id: str = ""

class ChatResponse(BaseModel):
    response: str
    success: bool
    session_id: str

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    try:
        response = run_agent(session_id, request.message)
        return ChatResponse(response=response, success=True, session_id=session_id)
    except Exception as e:
        return ChatResponse(
            response="Something went wrong. Please try again.",
            success=False,
            session_id=session_id
        )

@app.get("/health")
def health():
    return {"status": "AI Agent running", "tools": ["search", "news", "weather", "calculator", "code", "currency"]}