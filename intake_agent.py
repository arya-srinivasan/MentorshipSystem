#intake agent

"""
Intake Agent 

Conversation agent that chats with student to clarify their question. 
Keeps an updated summary in session state each time the student says something to the chat. 
Agent decides when to stop asking questions based on when it has enough context to exit. 
Outputs: {summarized_question, prior_knowledge}

- Google ADK (LoopAgent pattern)
"""
from dotenv import load_dotenv
load_dotenv()

import json
from google.adk.agents import LlmAgent, LoopAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import FunctionTool
from google.genai import types as genai_types

#swap this later with database session stuff
session_service = InMemorySessionService()
APP_NAME = "intake_agent"
MODEL = "gemini-2.5-flash"


# Tools

def update_summary(summarized_question: str, prior_knowledge: str, is_ready_to_classify: bool,) -> dict:
    """
    Called by the agent each time the student says something to the chat to update the understanding of the question. 
    Set is_ready_to_classify=True when the agent has enough context and the question is specific enough to hand off to classifier agent - exits the loop.

    Args:
        summarized_question: The best one-sentence summary of what the student is confsued about based on the conversation so far. 
        prior_knowledge: What the agent thinks the student already knows about the topic based on the conversation so far.
        is_ready_to_classify: Whether the agent thinks it has enough context to exit and hand off to the classifier agent. True when no more clarification is needed. 

    Returns:
        dict: A dictionary containing the updated summary, prior knowledge, and whether the agent is ready to classify.
    """
    return {
        "summarized_question": summarized_question,
        "prior_knowledge": prior_knowledge,
        "is_ready_to_classify": is_ready_to_classify
    }

update_summary_tool = FunctionTool(func=update_summary)

# System Prompt

INTAKE_SYSTEM_PROMPT = """
You are a warm, friendly, and patient academic advisor helping a student articulate their question before a lecture, office hours, or tutoring session. 

Your goal is to have a conversation with the student to clarify their question and gather context about what they already know so that their question can be routed to the right topic and asnwered as effectively as possible.

Behavior guidelines:
- Be warm, friendly, and patient. The student may be confused or frustrated, so it's important to be empathetic and encouraging.
- Ask ONE focused classifying question at a time
- If the student's question is too broad, ask them to narrow it down. 
- If the student's question is already clear, don't over-clarify - exit quickly

AFTER EVERY STUDENT MESSAGE you MUST follow this exact order:
  1. Call update_summary() to record your current understanding
  2. THEN write your reply to the student

RULES FOR update_summary():
  - summarized_question: your best current one-sentence summary of the confusion
  - prior_knowledge: what they seem to already understand (infer from how they talk)
  - is_ready_to_classify: True ONLY when ALL of these are true:
      1. The student has named a SPECIFIC topic or concept — NOT just a subject like "physics" or "math"
      2. You know exactly what confuses them about that specific topic
      3. You have asked AT LEAST 2 clarifying questions and received answers to both

CRITICAL: If is_ready_to_classify=True, do NOT ask another question. Just tell the student you have what you need.
CRITICAL: If is_ready_to_classify=False, you MUST ask exactly one clarifying question in your reply. Do not exit.
When to Exit:
    You decide when to exit based on specificity and clarity of the student's question. There is no fixed number of terns but exit as soon as you have enough, do not over-clarify.
   
- is_ready_to_classify: True ONLY when ALL of these are true:
    1. The student has named a SPECIFIC topic or concept (e.g. "Newton's laws", 
     "balancing equations", "integration by parts") — NOT just a subject like "physics"
    2. You understand what specifically confuses them about that topic
    3. You have a sense of what they already know
      
- NEVER set is_ready_to_classify=True if the student has only named a broad 
    subject like "physics", "math", "chemistry". You MUST keep asking until they 
    get specific. A subject name alone is never enough.

Style:
- short, friendly messages (like a real person/tutor)
- never ask multiple questions at once
- acknowledge what they say to show you are listening and understanding what they are saying
- when exiting, let the student know that you have everything you need to help them and that you will get back to them soon with an answer to their question.

"""

# Agents 

#Inner agent: does one turn of conversation with the student then calls update_summary 
intake_turn_agent = LlmAgent(
    name="intake_turn_agent",
    model=MODEL,
    instruction=INTAKE_SYSTEM_PROMPT,
    tools=[update_summary_tool],
    output_key="last_summary", #stores this response in sesion state
)

#Outer agent: calls the inner agent in a loop until it decides to exit based on is_ready_to_classify=True
intake_loop_agent = LoopAgent(
    name="intake_loop_agent",
    sub_agents=[intake_turn_agent], #exit when tool sets is_ready_to_classify=True
    max_iterations=10, #safety cap in case
)

# Runner
runner = Runner(agent=intake_loop_agent, app_name=APP_NAME, session_service=session_service)

def extract_found(events: list) -> dict | None:
    """
    Scans agent events for the last update_summary_tool call and extracts the arguments to check if is_ready_to_classify is True and returns the found question context.
    """
    found = None
    for event in events:
        if not hasattr(event, "content") or not event.content: #skip metadata events
            continue
        for part in event.content.parts or []:
            if hasattr(part, "function_response") and part.function_response: #only look at tool responses
                resp = part.function_response.response
                if isinstance(resp, dict) and resp.get("is_ready_to_classify"):
                    found = {
                        "summarized_question": resp.get("summarized_question", ""),
                        "prior_knowledge":     resp.get("prior_knowledge", ""),
                    }
    return found
    
async def run_intake_session(user_id: str, session_id: str, user_message: str):
    """
    Send one user message to the intake agent and return:
      - agent_reply: the text to show the student
      - found: non-None when the loop has exited and is ready for classifier
    """
    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_message)],
    )
 
    events = []
    agent_reply = ""
 
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        events.append(event)
        # Capture the last text response from the agent
        if (
            hasattr(event, "content")
            and event.content
            and event.content.role == "model"
        ):
            for part in event.content.parts or []:
                if hasattr(part, "text") and part.text:
                    agent_reply = part.text
 
    found = extract_found(events)
    return agent_reply, found
 


# ── CLI demo ──────────────────────────────────────────────────────────────────
 
async def demo():
    """
    Simple terminal demo. In production this is called by the coordinator.
    """
    import asyncio
 
    user_id    = "student_001"
    session_id = "session_001"
 
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )
 
    print("\n── Intake Agent Demo ──────────────────────────────")
    print("Type your question. The agent will clarify until it's ready.")
    print("Type 'quit' to exit.\n")
 
    first_message = input("You: ").strip()
    if first_message.lower() == "quit":
        return
 
    while True:
        reply, payload = await run_intake_session(user_id, session_id, first_message)
 
        print(f"\nAgent: {reply}\n")
 
        if payload:
            print("── Loop exited. Final payload for classifier ──")
            print(json.dumps(payload, indent=2))
            print("───────────────────────────────────────────────\n")
            break
 
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        first_message = user_input
 
 
if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())