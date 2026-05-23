import os
import uuid
from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent
from database.db import get_questions, mark_question_answered, add_question
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import asyncio
from google.genai.types import Content, Part

load_dotenv()

from classifier_agent import handle_student_question
from faculty_assistant import run_faculty_assistant

APP_NAME = "workflow"
USER_ID = "user_1"


runner = None
session_id = None


async def setup():
    global runner, session_id
    session_id      = f"events_{uuid.uuid4()}"
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    runner = Runner(
        app_name=APP_NAME,
        agent=workflow_agent,
        session_service=session_service
    )

async def main():
    await setup()

    response = await handle_student_question(
        conversation_id=session_id, 
        question="Can you explain what you meant by that last example?", 
        session_id=session_id, 
        user_id=USER_ID
    )
    print("Classifier Response:", response)

    faculty_response = await run_faculty_assistant(
        conversation_id=session_id, 
        session_id=session_id, 
        user_id=USER_ID
    )
    print("Faculty Assistant Response:", faculty_response)

if __name__ == "__main__":
    asyncio.run(main())