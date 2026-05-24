import os
import uuid
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from database.db import get_questions, mark_question_answered, add_question
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import asyncio
from google.genai.types import Content, Part

load_dotenv()


# First workflow
    # run intake session
    # run classifying session
    # call RAG agent

# Second workflow
    # run intake session
    # run classifying session
    # follow-up question workflow (calls RAG agent within)


# Main workflow
# Call first workflow 
# While true 
# call second workflow agent
