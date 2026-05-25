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



async def main():
    session_id = f"session_{uuid.uuid4()}"

    workflow_response = await handle_student_question(
        conversation_id=session_id, 
        question="Can you explain what you meant by that last example?", 
        session_id=session_id, 
        user_id=USER_ID
    )
    print("Classifier Response:", workflow_response)

if __name__ == "__main__":
    asyncio.run(main())