from curses import raw
import os
import json
import asyncio
from google.genai.types import Content, Part
import json
import asyncio
from google.genai.types import Content, Part
from google.adk.agents import LlmAgent
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from database.db import add_question, get_conversation_context
from relevant_transcript import meeting_copilot_agent
from faculty_assistant import run_faculty_assistant
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from database.db import add_question, get_conversation_context
from relevant_transcript import meeting_copilot_agent
from faculty_assistant import run_faculty_assistant

load_dotenv()

# extract text helper
def extract_text(r) -> str:
    if isinstance(r, str): return r
    if hasattr(r, "content"):
        try: return r.content.parts[0].text
        except: pass
    return str(r) if r else "No response."

question_classifier = LlmAgent(
    name="Question_Classifier",
    model="gemini-2.5-flash",
    description="Classifies whether the user question can be answered by the agent or not.",
    instruction="""
    You are a classifier for a student Q&A system used during live lectures.

    Your job is to decide whether a student's question can be answered by the AI agent, 
    or whether it requires the faculty member's attention.

    RULES - The agent CAN answer questions that are:
    - Factual and grounded in general course knowledge or provided materials
    - Conceptual clarifications that don't depend on what was just said in the lecture
    - Common questions with clear, well-established answers (definitions, formulas, processes)

    RULES - The agent CANNOT answer questions that:
    - Reference something specific the faculty just said or showed (e.g. "what did you mean by that?")
    - Require the faculty's personal opinion, judgment, or teaching intent
    - Are ambiguous and could be misinterpreted without more context from the live session
    - Are about logistics only the faculty would know (deadlines, grading, expectations)

    OUTPUT FORMAT:
    Respond only with a JSON object. No explanation, no extra text, no code fences.

    {
    "decision": "agent" | "faculty",
    "confidence": 0.0 - 1.0,
    "reason": "one sentence explaining the decision"
    }
    """,
    output_key="decision",
)


session_service = InMemorySessionService()
runner = Runner(
    agent=question_classifier, 
    app_name="Question Classifier", 
    session_service=session_service,
)

async def handle_student_question(conversation_id, question, session_id, user_id):
    try:
        await session_service.create_session(
        app_name="Question Classifier", user_id=user_id, session_id=session_id
    )
    except Exception:
        pass

    result = runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=Content(role="user", parts=[Part(text=question)])
    )
    for event in result:
        if event.is_final_response():
            try:
                #decision = json.loads(event.content.parts[0].text)
                raw = event.content.parts[0].text
                raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                decision = json.loads(raw)

                add_question(conversation_id, question)

                if decision["decision"] == "faculty":
                    faculty_response = await run_faculty_assistant(conversation_id=session_id, session_id=session_id, user_id=user_id, question=question, context=get_conversation_context(conversation_id, question))
                    return faculty_response
                else:
                    # run meeting_copilot_agent here
                    from relevant_transcript import run_transcript
                    response = await run_transcript(user_id, session_id)
                    return extract_text(response) if response else "Question answered by agent."
            except json.JSONDecodeError:
                return "Classifier returned an unexpected response."
        
    return "No response from classifier."
