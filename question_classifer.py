import os
import json
import asyncio
from google.genai.types import Content, Part
from google.adk.agents import LlmAgent
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from database.db import add_question
from relevant_transcript import meeting_copilot_agent
from faculty_assistant import run_faculty_assistant

load_dotenv()

question_classifier = LlmAgent(
    name="Question Classifier",
    model="gemini-2.0-flash",
    description="Classifies whether the user question can be answered by the agent or not.",
    instructions="""
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
    Respond only with a JSON object. No explanation, no extra text.

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
    await session_service.create_session(
        app_name="Question Classifier", 
        user_id=user_id,
        session_id=session_id
    )
    result = runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=Content(role="user", parts=[Part(text=question)])
    )
    for event in result:
        if event.is_final_response():
            try:
                decision = json.loads(event.content.parts[0].text)

                if decision["decision"] == "faculty":
                    add_question(conversation_id, question)
                    faculty_response = await run_faculty_assistant(conversation_id=session_id, session_id=session_id, user_id=user_id)
                    return faculty_response
                else:
                    # run meeting_copilot_agent here
                    return "Question answered by agent."
            except json.JSONDecodeError:
                return "Classifier returned an unexpected response."
    return "No response from classifier."
