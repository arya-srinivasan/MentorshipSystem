import os
from google.adk.agents import LlmAgent
from dotenv import load_dotenv
from database.db import get_questions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

load_dotenv()

faculty_assistant = LlmAgent(
    name="Faculty Assistant",
    model="gemini-2.0-flash",
    description="An assistant for faculty members during live lectures to help manage student questions.",
    instructions="""
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
    tools=[get_questions],
    output_key="response",
)

session_service = InMemorySessionService()
runner = Runner (
    agent=faculty_assistant, 
    app_name="Faculty Assistant", 
    session_service=session_service,
)

async def run_faculty_assistant(conversation_id, session_id, user_id):
    result = runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=Content(role="user", parts=[Part(text="Check pending questions")])
    )

    for event in result:
        if event.is_final_response():
            return event.content.parts[0].text
    return "No response from faculty assistant."
