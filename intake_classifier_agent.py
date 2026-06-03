"""
intake_classifier_agent.py

Combined intake + classifier agent.

Replaces intake_agent.py and classifier_agent.py.

- Chats with student to clarify their question
- Once specific enough, classifies into a topic cluster in the same turn
- Exits and returns { summarized_question, prior_knowledge, topic_cluster, confidence }

Google ADK (LoopAgent pattern)
"""

from dotenv import load_dotenv
load_dotenv()

import json
from google.adk.agents import LlmAgent, LoopAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import FunctionTool
from google.genai import types as genai_types

session_service = InMemorySessionService()
APP_NAME = "intake_classifier_agent"
MODEL    = "gemini-2.5-flash"



# ── Topic clusters 

TOPIC_CLUSTERS = [
    "Arrays & Strings",
    "Recursion",
    "Sorting & Searching",
    "Linked Lists",
    "Stacks & Queues",
    "Trees & Binary Search Trees",
    "Graphs & Graph Traversal",
    "Dynamic Programming",
    "Hashing & Hash Maps",
    "Time & Space Complexity",
    "Object-Oriented Design",
    "Other / General",
]

EXAMPLE_SYLLABUS = """
Course: CS 101 — Introduction to Computer Science
Topics covered (in order):
  Week 1-2:  Arrays, Strings, Basic I/O
  Week 3-4:  Recursion, Call Stack
  Week 5-6:  Sorting (bubble, merge, quick), Binary Search
  Week 7-8:  Linked Lists, Stacks, Queues
  Week 9-10: Trees, BSTs, Tree Traversal
  Week 11-12: Graphs, BFS, DFS
  Week 13-14: Dynamic Programming, Memoization
  Week 15:   Review & Final Exam Prep
"""

# ── Tool 

def submit_question(
    summarized_question: str,
    prior_knowledge: str,
    topic_cluster: str,
    confidence: str,
    is_ready: bool,
) -> dict:
    """
    Called by the agent after every student message to record current understanding.
    When the question is specific enough AND classified, set is_ready=True to exit.

    Args:
        summarized_question: Best one-sentence summary of what the student is confused about.
        prior_knowledge:     What the student already understands, inferred from conversation.
        topic_cluster:       One of the predefined topic cluster names. Empty string if not ready yet.
        confidence:          "high", "medium", or "low". Empty string if not ready yet.
        is_ready:            True when the question is specific enough AND classified. Exits the loop.

    Returns:
        Stored dict.
    """
    return {
        "summarized_question": summarized_question,
        "prior_knowledge":     prior_knowledge,
        "topic_cluster":       topic_cluster,
        "confidence":          confidence,
        "is_ready":            is_ready,
    }

submit_question_tool = FunctionTool(func=submit_question)

# ── System prompt 

SYSTEM_PROMPT = f"""
You are a warm, friendly academic assistant helping a student clarify and file their question before a lecture, office hours or study session.

Your two jobs in one conversation:
1. Clarify the student's question through back-and-forth dialogue
2. Once clear enough, classify it into a topic cluster and file it

COURSE SYLLABUS:
{EXAMPLE_SYLLABUS}

AVAILABLE TOPIC CLUSTERS (pick exactly one when ready):
{chr(10).join(f'  • {t}' for t in TOPIC_CLUSTERS)}

FEW-SHOT EXAMPLES:
  "I'm confused about sorting" → too vague → ask more  
  "I don't understand why merge sort's merge step is O(n)" → specific enough → classify as "Sorting & Searching"

AFTER EVERY STUDENT MESSAGE you MUST:
  1. Call submit_question() to record your understanding
  2. Write your reply to the student

RULES FOR submit_question():
  - Always fill in summarized_question and prior_knowledge with your best current understanding
  - Set topic_cluster and confidence only when is_ready=True, otherwise use empty strings
  - Set is_ready=True ONLY when ALL of these are true:
      1. Student named a SPECIFIC concept — not just a broad subject like "math", "CS", or "Python"
      2. You understand exactly what confuses them about it
      3. You have asked AT LEAST 2 clarifying questions and received answers to both
      4. You can confidently pick a topic cluster that is NOT "Other / General" — if you'd classify as Other/General, keep asking

CRITICAL RULES — NEVER BREAK THESE:
  - NEVER set is_ready=True on the first student message, no matter how specific it seems
  - NEVER set is_ready=True if you have asked fewer than 2 clarifying questions
  - NEVER classify as "Other / General" — if you would, ask another clarifying question instead
  - You MUST ask at least 2 questions before filing ANY question

WHEN is_ready=True:
  - Pick the best topic cluster from the list
  - Tell the student warmly: "Got it! I've filed your question under [Topic]. We'll get back to you when it comes up in the lecture."
  - Do NOT ask any more questions

WHEN is_ready=False:
  - Ask exactly ONE clarifying question
  - Be warm and encouraging
  - Never list multiple questions at once

STYLE:
  - Short, friendly messages like a real tutor
  - Acknowledge what they say before asking follow-up
  - Never show the student the topic cluster name until you're filing it
"""

#  Agents 

intake_classifier_turn_agent = LlmAgent(
    name="intake_classifier_turn_agent",
    model=MODEL,
    instruction=SYSTEM_PROMPT,
    tools=[submit_question_tool],
    output_key="last_output",
)

intake_classifier_loop_agent = LoopAgent(
    name="intake_classifier_loop_agent",
    sub_agents=[intake_classifier_turn_agent],
    max_iterations=10,
)

#  Runner 

runner = Runner(
    agent=intake_classifier_loop_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

#  Helpers 

def extract_result(events: list) -> dict | None:
    """
    Scans events for the last submit_question call where is_ready=True.
    Returns the full result dict or None if not ready yet.
    """
    
    result = None
    for event in events:
        if not hasattr(event, "content") or not event.content:
            continue
        for part in event.content.parts or []:
            if hasattr(part, "function_response") and part.function_response:
                resp = part.function_response.response
                if isinstance(resp, dict) and resp.get("is_ready"):
                    result = {
                        "summarized_question": resp.get("summarized_question", ""),
                        "prior_knowledge":     resp.get("prior_knowledge", ""),
                        "topic_cluster":       resp.get("topic_cluster", "Other / General"),
                        "confidence":          resp.get("confidence", "low"),
                    }
    return result


async def run_intake_classifier_session(
    user_id: str,
    session_id: str,
    user_message: str,
) -> tuple[str, dict | None]:
    """
    Send one user message to the combined agent.
    Returns (agent_reply, result).
    result is non-None when the agent has clarified and classified — ready for DB.
    """
    try:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
    except Exception:
        pass

    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_message)],
    )

    events     = []
    agent_reply = ""

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        events.append(event)
        if (
            hasattr(event, "content")
            and event.content
            and event.content.role == "model"
        ):
            for part in event.content.parts or []:
                if hasattr(part, "text") and part.text:
                    agent_reply = part.text

    result = extract_result(events)
    return agent_reply, result


# ── CLI demo ──────────────────────────────────────────────────────────────────

async def demo():
    import asyncio

    user_id    = "student_001"
    session_id = "demo_session_001"

    print("\n── Intake + Classifier Demo ───────────────────────")
    print("Type your question. Agent clarifies then classifies.")
    print("Type 'quit' to exit.\n")

    first_message = input("You: ").strip()
    if first_message.lower() == "quit":
        return

    while True:
        reply, result = await run_intake_classifier_session(user_id, session_id, first_message)
        print(f"\nAgent: {reply}\n")

        if result:
            print("── Done. Result for DB ──")
            print(json.dumps(result, indent=2))
            print("────────────────────────\n")
            break

        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        first_message = user_input


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())