import os
from google.adk.agents import LlmAgent
from dotenv import load_dotenv
from database.db import get_questions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

load_dotenv()
from google.adk.tools import FunctionTool

def make_get_questions(conversation_id):
    def get_pending_questions() -> list:
        """Returns all pending student questions for the current session."""
        return get_questions(conversation_id=conversation_id)
    return get_pending_questions

def agent(conversation_id):
    faculty_assistant = LlmAgent(
        name="Faculty_Assistant", 
        model="gemini-2.5-flash",
        description="An assistant for faculty members during live lectures.",
        instruction="""
        You are a real-time assistant for faculty members during live lectures. Your sole job is to monitor 
        incoming student questions and surface the most important ones to the faculty member at the right moment — 
        without being disruptive or overwhelming.

        RESPONSIBILITIES:
        - Continuously monitor the queue of pending student questions using your tools
        - Prioritize questions that are frequently asked by multiple students over one-off questions
        - Surface questions clearly and concisely so faculty can act on them quickly mid-lecture

        HOW TO SURFACE QUESTIONS:
        - Be brief — faculty are mid-lecture and have limited attention
        - Lead with the most important question first
        - If multiple students asked the same thing, group them as one item and note the count
        - Always include a suggested talking point or answer to help faculty respond quickly

        OUTPUT FORMAT:
        When surfacing questions, always follow this structure:
        
        [Question count if multiple] Student Question: <cleaned question>
        Suggested Response: <brief suggested answer or talking point>
        
        RULES:
        - Never surface questions that have already been answered or dismissed
        - Never overwhelm faculty with more than 3 questions at a time
        - If no questions are pending, respond with: "No pending questions at the moment."
        - Do not editorialize or add unnecessary commentary — keep it tight and actionable
        """,
        tools=[FunctionTool(func=make_get_questions(conversation_id))],  # ← callable
        output_key="response",
    )
    return faculty_assistant


async def run_faculty_assistant(conversation_id, session_id, user_id, question):
    a = agent(conversation_id)
    session_service = InMemorySessionService()
    runner = Runner(
        agent=a,
        app_name="Faculty_Assistant",
        session_service=session_service,
    )

    try:
        await session_service.create_session(
            app_name="Faculty_Assistant",
            user_id=user_id,
            session_id=session_id,
        )
    except Exception:
        pass 

    msg = f"Initial user response: {question}"

    result = runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=Content(role="user", parts=[Part(text=msg)])
    )

    for event in result:
        if event.is_final_response():
            return event.content.parts[0].text

    return "No response from faculty assistant."