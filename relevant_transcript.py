"""
relevant_transcript.py

Real-time meeting copilot:
- Embeds incoming transcript chunks and stores them in Pinecone (so they
  can be retrieved as "meeting memory" later).
- Runs a retrieval-augmented LLM agent (meeting_copilot_agent) on each
  chunk to surface insights / action items / relevant context.

This is a cleaned-up, importable version of the original Colab export.
All the Colab-only bits (getpass, google.colab.userdata, top-level
pc.create_index() side effects that require a key at import time) have
been removed so this module can be safely imported from backend.py,
final_worflow.py, question_classifer.py, etc. without crashing when
PINECONE_API_KEY / GROQ_API_KEY aren't set (e.g. during local dev or
when running the static-transcript test).
"""

import os
import time
import uuid

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.tools import FunctionTool
from google.genai.types import Content, Part
from google.genai import types

load_dotenv()

APP_NAME = "relevant_transcript"
INDEX_NAME = "meeting-memory"
NAMESPACE = "meeting_042"
EMBED_DIM = 384


# ---------------------------------------------------------------------------
# Pinecone setup (lazy / optional - won't crash if no API key is configured)
# ---------------------------------------------------------------------------

_pc = None
_index = None


def _get_index():
    """Lazily create the Pinecone client + index on first use."""
    global _pc, _index

    if _index is not None:
        return _index

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("[relevant_transcript] PINECONE_API_KEY not set - vector storage disabled")
        return None

    from pinecone import Pinecone, ServerlessSpec

    _pc = Pinecone(api_key=api_key)

    existing = [i["name"] for i in _pc.list_indexes()]
    if INDEX_NAME not in existing:
        _pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    _index = _pc.Index(INDEX_NAME)
    return _index


# ---------------------------------------------------------------------------
# Embedding model (lazy - first call downloads/loads the model)
# ---------------------------------------------------------------------------

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


# ---------------------------------------------------------------------------
# Storage + retrieval
# ---------------------------------------------------------------------------

def store_chunk(chunk_text: str, chunk_id: str = None, metadata: dict = None) -> None:
    """Embed a transcript chunk and upsert it into Pinecone."""
    index = _get_index()
    if index is None:
        return

    chunk_id = chunk_id or f"chunk_{uuid.uuid4()}"
    embedding = _get_embed_model().encode(chunk_text).tolist()

    meta = {"text": chunk_text, "timestamp": time.time()}
    if metadata:
        meta.update(metadata)

    index.upsert(
        vectors=[{"id": chunk_id, "values": embedding, "metadata": meta}],
        namespace=NAMESPACE,
    )


def retrieve_relevant_transcript(query: str) -> str:
    """Retrieve relevant transcript chunks based on cosine similarity of embeddings."""
    index = _get_index()
    if index is None:
        return ""

    query_embedding = _get_embed_model().encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True,
        namespace=NAMESPACE,
    )

    return "\n".join(match["metadata"]["text"] for match in results["matches"])


retrieval_tool = FunctionTool(retrieve_relevant_transcript)


# ---------------------------------------------------------------------------
# Meeting copilot agent
# ---------------------------------------------------------------------------

meeting_copilot_agent = Agent(
    name="meeting_copilot_agent",
    model="gemini-2.5-flash",
    tools=[retrieval_tool],
    instruction="""
You are a real-time AI meeting copilot.

Your job is to:
- analyze live transcript chunks
- retrieve relevant meeting memory
- identify important insights
- detect action items
- surface relevant context
- keep responses concise and useful

When analyzing a transcript chunk:
1. Use the retrieve_relevant_transcript tool
2. Find relevant historical context
3. Generate concise real-time assistance

Focus on:
[Summary of relevant context]
""",
)

session_service = InMemorySessionService()
runner = Runner(
    agent=meeting_copilot_agent,
    app_name=APP_NAME,
    session_service=session_service,
)


def _extract_text(event) -> str:
    if hasattr(event, "content") and event.content and event.content.parts:
        return "".join(part.text for part in event.content.parts if hasattr(part, "text") and part.text)
    return ""


async def analyze_transcript_chunk(
    text: str,
    user_id: str = "zoom_bot",
    session_id: str = "live_session",
    chunk_id: str = None,
    metadata: dict = None,
) -> str:
    """
    Main entry point for the live pipeline.

    Stores the chunk in Pinecone (meeting memory) and runs the
    meeting_copilot_agent on it, returning whatever insight/response
    the agent generates.
    """
    store_chunk(text, chunk_id=chunk_id, metadata=metadata)

    try:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
    except Exception:
        # session already exists - fine
        pass

    return f"Chunk stored: {len(text.split())} words"


# ---------------------------------------------------------------------------
# Backwards-compatible demo entry point
#
# final_worflow.py / question_classifer.py call run_transcript(user_id, session_id)
# with no chunk text - kept here so those imports don't break. It just runs
# the copilot agent over a small demo stream.
# ---------------------------------------------------------------------------

_DEMO_TRANSCRIPT_STREAM = [
    "AWS SageMaker costs are growing rapidly.",
    "GPU inference optimization might help.",
    "We should analyze EC2 spot instances.",
]

async def run_transcript(user_id: str, session_id: str, question: str = None) -> str:
    """Query Pinecone with the student's question and return an answer."""
    try:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
    except Exception:
        pass

    prompt = question if question else "Summarize the key points discussed so far."

    content = Content(role="user", parts=[Part(text=prompt)])
    final_response = ""
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            final_response = _extract_text(event)
    return final_response
