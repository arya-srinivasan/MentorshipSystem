# api.py

"""
endpoint descriptions: 
POST /chat — student sends a message
POST /transcript/chunk — Prithika sends transcript chunks
POST /transcript/answer — Pariya writes answers back
GET  /answered/{session_id} — frontend polls for answered questions
GET  /questions/{session_id} — get waiting questions
GET  /health — check if server is running

"""
from dotenv import load_dotenv
load_dotenv()

import os, sys, uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

#load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from intake_agent import run_intake_session
# from classifier_agent import classify_student_question
from intake_classifier_agent import run_intake_classifier_session
from relevant_transcript import run_transcript
from question_classifer import handle_student_question
from database.db import add_question, get_questions, create_table, get_answered_questions, mark_question_answered

from contextlib import asynccontextmanager

transcript_chunks = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_table()
    yield

app = FastAPI()
from fastapi.staticfiles import StaticFiles
app.mount("/js", StaticFiles(directory="src"), name="js")
app.mount("/css", StaticFiles(directory="src"), name="css")

#app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

sessions: dict = {}
classifier_results: dict = {}
session_phase: dict = {}

class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    path: str
    topic_cluster: Optional[str] = None
    summarized_question: Optional[str] = None

@app.on_event("startup")
async def startup():
    create_table()

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    is_first = req.user_id not in sessions
    session_id = sessions.setdefault(req.user_id, f"session_{uuid.uuid4()}")

    if is_first or session_phase.get(session_id) == "intake":
        reply, result = await run_intake_classifier_session(req.user_id, session_id, req.message)

        if not result:
            session_phase[session_id] = "intake"
            return ChatResponse(response=reply, session_id=session_id, path="intake", topic_cluster=None)

        session_phase[session_id] = "classified"
        classifier_results[session_id] = result
        add_question(session_id, result["summarized_question"], result["topic_cluster"])

        return ChatResponse(
            response=reply,
            session_id=session_id,
            path="ai",
            topic_cluster=result["topic_cluster"],
            summarized_question=result["summarized_question"],
        )

    else:
        prev = classifier_results.get(session_id, {})
        response = await handle_student_question(session_id, req.message, session_id, req.user_id)
        text = extract_text(response)
        return ChatResponse(
            response=text,
            session_id=session_id,
            path="faculty" if "Student Question:" in text else "ai",
            topic_cluster=prev.get("topic_cluster")
        )
    
@app.get("/questions/{conversation_id}")
def questions(conversation_id: str):
    return {"questions": get_questions(conversation_id)}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return FileResponse("src/index.html")

@app.get("/answered/{session_id}")
def answered(session_id: str):
    return {"questions": get_answered_questions(session_id)}

def extract_text(r) -> str:
    if isinstance(r, str): return r
    if hasattr(r, "content"):
        try: return r.content.parts[0].text
        except: pass
    return str(r) if r else "No response."

class TranscriptChunk(BaseModel):
    text: str
    timestamp: Optional[str] = None

@app.post("/transcript/chunk")
async def recieve_chunk(chunk: TranscriptChunk):
    #relevant transcript agent plug in here
    transcript_chunks.append({"text": chunk.text, "timestamp": chunk.timestamp})
    return {"status": "received"}

class AnswerUpdate(BaseModel):
    conversation_id: str
    question: str
    answer: str

@app.post("/transcript/answer")
def receive_answer(update: AnswerUpdate):
    mark_question_answered(update.conversation_id, update.question, update.answer)
    return {"status": "updated"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)