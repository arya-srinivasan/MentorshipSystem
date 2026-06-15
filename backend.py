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
import asyncio
import json
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
from relevant_transcript import analyze_transcript_chunk, run_transcript
from question_classifer import handle_student_question
from database.db import add_question, get_questions, create_table, get_answered_questions, mark_question_answered

from contextlib import asynccontextmanager

transcript_chunks = []
sse_queues: dict = {}
meeting_active = False

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
    global meeting_active
    print(f"Meeting Active: {meeting_active}")

    #is_first = req.user_id not in sessions
    session_id = sessions.setdefault(req.user_id, f"session_{uuid.uuid4()}")

    if meeting_active:
        print("Start handle")
        response = await handle_student_question(session_id, req.message, session_id, req.user_id)
        text = extract_text(response)
        return ChatResponse(
            response=text,
            session_id=session_id,
            path="faculty" if "Student Question:" in text else "ai",
            topic_cluster=classifier_results.get(session_id, {}).get("topic_cluster"),  # ← fixed
        )

    reply, result = await run_intake_classifier_session(req.user_id, session_id, req.message)
    if not result:
        session_phase[session_id] = "intake"
        return ChatResponse(response=reply, session_id=session_id, path="intake")

    classifier_results[session_id] = result
    add_question(session_id, result["summarized_question"], result["topic_cluster"])
    
    session_phase[session_id] = "classified"

    return ChatResponse(
        response=reply,
        session_id=session_id,
        path="classified",
        topic_cluster=result["topic_cluster"],
        summarized_question=result["summarized_question"],
    )
    
@app.get("/questions/{conversation_id}")
def questions(conversation_id: str):
    return {"questions": get_questions(conversation_id=conversation_id)}

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
    speaker: Optional[str] = None
    session_id: Optional[str] = "live_session"

@app.post("/transcript/chunk")
async def receive_chunk(chunk: TranscriptChunk):
    global meeting_active

    transcript_chunks.append({
        "text": chunk.text,
        "timestamp": chunk.timestamp,
        "speaker": chunk.speaker,
    })

    if not meeting_active:
        meeting_active = True
        task = asyncio.create_task(answer_queued(chunk.session_id))

    try:
        insight = await analyze_transcript_chunk(
            chunk.text,
            session_id=chunk.session_id,
            metadata={"timestamp": chunk.timestamp, "speaker": chunk.speaker},
        )
    except Exception as e:
        print(f"[transcript/chunk ERROR] {e}")
        raise

    return {"status": "received", "insight": insight}

@app.get("/transcript/chunks")
def list_chunks():
    """Debug endpoint - see everything received so far."""
    return {"chunks": transcript_chunks}

class AnswerUpdate(BaseModel):
    conversation_id: str
    question: str
    answer: str

@app.post("/transcript/answer")
def receive_answer(update: AnswerUpdate):
    mark_question_answered(update.conversation_id, update.question, update.answer)
    return {"status": "updated"}

async def answer_queued(session_id: str):
    while meeting_active:
        try:
            queued = get_questions()
            for q in queued:
                try:
                    response = await run_transcript("system", session_id, question=q["question"])
                    text = extract_text(response)
                    mark_question_answered(q["conversation_id"], q["question"], text)
                
                except Exception as e:
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            import traceback
            traceback.print_exc()
        await asyncio.sleep(10)
    print("[queued] loop stopped")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
