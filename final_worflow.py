import os
import uuid
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from database.db import get_questions, mark_question_answered, add_question
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import asyncio
from google.genai.types import Content, Part
from intake_agent import run_intake_session
from classifier_agent import classify_student_question
from relevant_transcript import run_transcript
from question_classifer import handle_student_question

load_dotenv()



# First workflow
async def start_conversation(user_id, session_id, first_message):
    reply, payload = await run_intake_session(user_id, session_id, first_message)

    print(reply)

    await classify_student_question(
        intake_payload=payload,
        user_id=user_id,
        session_id=session_id,
    )

    final_response = await run_transcript(user_id, session_id)

    return final_response


# Second workflow
async def main_conversation_loop(user_id, session_id, first_message):
    # run intake session
    reply, payload = await run_intake_session(user_id, session_id, first_message)

    print(reply)
    # run classifying session
    await classify_student_question(
        intake_payload=payload,
        user_id=user_id,
        session_id=session_id,
    )

    # follow-up question workflow (calls RAG agent within)
    final_response = await handle_student_question(conversation_id=session_id, question=payload, session_id=session_id, user_id=user_id)

    return final_response

USER_ID = "user_1"

# Main workflow
async def main():
    session_id = f"session_{uuid.uuid4()}"

    first_message = input("User: ").strip()

    first_response = await start_conversation(USER_ID, session_id, first_message)
    print("Agent:", first_response)

    while True:
        message = input("Student: ").strip()

        if message.lower() in ["exit", "quit", "bye"]:
            print("Session ended.")
            break
        response = await main_conversation_loop(USER_ID, session_id, first_message)
        print("Agent:", response)


if __name__ == "__main__":
    asyncio.run(main())

