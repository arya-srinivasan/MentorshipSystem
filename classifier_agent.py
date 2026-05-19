"""
Classifier Agent

Takes the intake agent's output + course syllabus/ topics: 
- assigns the question to a predefined topic cluster and if topic is not covered in given topics, assign it to "other"
- confirm cluster with the student
- if student disagrees, re-classify once with their feedback
- outputs: {topic_cluster, confidence, summarized_question, prior_knowledge} → hands off this to coordinator

"""

import json
from google.adk.agents import LlmAgent, LoopAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import FunctionTool
from google.genai import types as genai_types

#session service
session_service = InMemorySessionService()
APP_NAME = "classifier_agent"
MODEL = "gemini-2.5-flash"

#predefined topic clusters
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

#course syllabus? 
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

#few-shot 
FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
  summarized_question: "I understand how merge sort splits arrays but I don't
                        see why the merge step is O(n) and not O(n log n)"
  prior_knowledge: "Knows recursion basics and Big-O notation"
  → topic_cluster: "Sorting & Searching"
  → confidence: high
  → reasoning: Question is specifically about merge sort's merge step complexity
 
EXAMPLE 2:
  summarized_question: "I don't understand why my recursive function keeps
                        hitting a stack overflow even with a base case"
  prior_knowledge: "Understands what a base case is, new to recursion"
  → topic_cluster: "Recursion"
  → confidence: high
  → reasoning: Stack overflow in recursion = call stack issue, core recursion topic
 
EXAMPLE 3:
  summarized_question: "I'm confused about why we use a visited set in BFS
                        but the textbook example doesn't show one"
  prior_knowledge: "Has seen BFS pseudocode, understands queues"
  → topic_cluster: "Graphs & Graph Traversal"
  → confidence: high
  → reasoning: BFS visited tracking is a graph traversal implementation detail
 
EXAMPLE 4:
  summarized_question: "I don't understand when to use a hash map vs a list
                        for storing data"
  prior_knowledge: "Knows both exist, unclear on tradeoffs"
  → topic_cluster: "Hashing & Hash Maps"
  → confidence: medium
  → reasoning: Tradeoff question — could touch arrays too, but hash map is primary
 
EXAMPLE 5:
  summarized_question: "Why does the professor keep saying 'n log n is better
                        than n squared' — what does that actually mean?"
  prior_knowledge: "No prior Big-O experience"
  → topic_cluster: "Time & Space Complexity"
  → confidence: high
  → reasoning: Core Big-O conceptual question, not tied to a specific algorithm
"""

#tools
def assign_cluster(topic_cluster: str, confidence: str, reasoning: str, is_confirmed: bool,) -> dict:
    """
    Called by the classifier agent to record its topic assignment.
    Call with is_confirmed=False on first classification (before student confirms).
    Call with is_confirmed=True once the student agrees, to exit the loop.
 
    Args:
        topic_cluster:  One of the predefined topic cluster names.
        confidence:     "high", "medium", or "low".
        reasoning:      One sentence explaining why this cluster was chosen.
        is_confirmed:   False = propose to student. True = student agreed, exit loop.
 
    Returns:
        Stored classification dict.
    """

    return {
        "topic_cluster": topic_cluster,
        "confidence": confidence,
        "reasoning": reasoning,
        "is_confirmed": is_confirmed
    }

assign_cluster_tool = FunctionTool(func=assign_cluster)

#System prompt

def build_classifier_prompt(payload: dict, syllabus: str) -> str:
    return f"""
    You are an expert academic classifier. Your job is to assign a student's question to the most relevant topic cluster so it can be routed to the right discussion group.


    COURSE SYLLABUS:
    {syllabus}

    AVAILABLE TOPIC CLUSTERS (you must pick exactly one):
    {chr(10).join(f'  • {t}' for t in TOPIC_CLUSTERS)}
 

    FEW-SHOT CLASSIFICATION EXAMPLES:
    {FEW_SHOT_EXAMPLES}

 
    STUDENT'S QUESTION PAYLOAD:
    summarized_question: {payload.get('summarized_question', '')}
    prior_knowledge:     {payload.get('prior_knowledge', '')}
 

 
    YOUR Step by step process:
 
    STEP 1 — CLASSIFY:
    - Pick the single best topic cluster from the predefined list
    - Call assign_cluster() with is_confirmed=False
    - Then present your classification to the student in a friendly way:
          "Based on your question, I'd put this under **[Topic]** — [one sentence why]. Does that sound right to you?"
 
    STEP 2 — WAIT FOR STUDENT RESPONSE:
    - If they say YES / correct / sounds good → call assign_cluster() again
        with the same cluster and is_confirmed=True. Tell them they're all set.
    - If they say NO / that's wrong → ask them which cluster sounds closer,
        or what aspect they think is being missed. Re-classify once with their
        feedback, then call assign_cluster() with is_confirmed=True on the
        updated cluster. Don't go back and forth more than once.
 
    RULES:
    - Always pick from the predefined list — never invent a new cluster name unless it's truly not covered, then use "Other / General"
    - If genuinely ambiguous, pick the more specific cluster and note it
    - Keep your message to the student short and friendly
    - Never show the student the internal reasoning or confidence score
    """
 
 
#
# 
 
def build_classifier_loop(payload: dict, syllabus: str = EXAMPLE_SYLLABUS) -> LoopAgent:
    """
    Builds the classifier loop agent with the payload baked into the prompt.
    Called fresh per session so each student gets the right context.
    """
    classifier_turn_agent = LlmAgent(
        name="classifier_turn_agent",
        model=MODEL,
        instruction=build_classifier_prompt(payload, syllabus),
        tools=[assign_cluster_tool],
        output_key="classifier_last_output",
    )
 
    return LoopAgent(
        name="classifier_loop_agent",
        sub_agents=[classifier_turn_agent],
        max_iterations=6,  # classify → confirm → (optional re-classify) → done
    )
 
 
#runner helpers
#  
def extract_confirmed_cluster(events: list) -> dict | None:
    """
    Scans events for the last assign_cluster call where is_confirmed=True.
    Returns the final classification payload or None if not yet confirmed.
    """
    result = None
    for event in events:
        if not hasattr(event, "content") or not event.content:
            continue
        for part in event.content.parts or []:
            if hasattr(part, "function_response") and part.function_response:
                resp = part.function_response.response
                if isinstance(resp, dict) and resp.get("is_confirmed"):
                    result = {
                        "topic_cluster": resp.get("topic_cluster", ""),
                        "confidence":    resp.get("confidence", ""),
                        "reasoning":     resp.get("reasoning", ""),
                    }
    return result
 
 
async def run_classifier_session(
    runner: Runner,
    user_id: str,
    session_id: str,
    user_message: str,
) -> tuple[str, dict | None]:
    """
    Send one message turn to the classifier agent.
    Returns (agent_reply, final_payload).
    final_payload is non-None when the student has confirmed the cluster.
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
        if (
            hasattr(event, "content")
            and event.content
            and event.content.role == "model"
        ):
            for part in event.content.parts or []:
                if hasattr(part, "text") and part.text:
                    agent_reply = part.text
 
    final_payload = extract_confirmed_cluster(events)
    return agent_reply, final_payload
 
 
# ── Full pipeline: intake payload → classifier → coordinator output ────────────
 
async def classify_student_question(
    intake_payload: dict,
    user_id: str,
    session_id: str,
    syllabus: str = EXAMPLE_SYLLABUS,
) -> dict:
    """
    Top-level function called by the coordinator after intake agent exits.
 
    Runs the full classifier conversation loop until the student confirms
    their topic cluster.
 
    Returns the final coordinator-ready payload:
    {
        "topic_cluster": str,
        "confidence": str,
        "summarized_question": str,
        "prior_knowledge": str,
    }
    """
    loop_agent = build_classifier_loop(intake_payload, syllabus)
 
    runner = Runner(
        agent=loop_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
 
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )
 
    # Kick off with the intake payload as the first "user" message
    # (coordinator passes this in; in production the UI triggers this)
    trigger_message = (
        f"Here is my question context:\n{json.dumps(intake_payload, indent=2)}"
    )
 
    print("\n── Classifier Agent ────────────────────────────────")
 
    first_reply, payload = await run_classifier_session(
        runner, user_id, session_id, trigger_message
    )
    print(f"\nAgent: {first_reply}\n")
 
    if payload:
        # Confirmed on first try (rare but possible)
        return {**payload, **intake_payload}
 
    # Confirmation loop
    while True:
        student_input = input("You: ").strip()
        reply, payload = await run_classifier_session(
            runner, user_id, session_id, student_input
        )
        print(f"\nAgent: {reply}\n")
 
        if payload:
            final = {**payload, **intake_payload}
            print("── Cluster confirmed. Final coordinator payload ──")
            print(json.dumps(final, indent=2))
            print("─────────────────────────────────────────────────\n")
            return final
 
 
# ── CLI demo ──────────────────────────────────────────────────────────────────
 
async def demo():
    """
    Simulates receiving a payload from the intake agent and running classification.
    """
    # Example payload as if it came from the intake agent
    example_intake_payload = {
        "summarized_question": (
            "I understand how merge sort splits the array recursively, "
            "but I don't understand why the merge step is O(n) — "
            "shouldn't comparing elements take longer?"
        ),
        "prior_knowledge": (
            "Understands recursion and what Big-O notation means, "
            "has seen merge sort pseudocode but hasn't implemented it."
        ),
    }
 
    await classify_student_question(
        intake_payload=example_intake_payload,
        user_id="student_001",
        session_id="classifier_session_001",
    )
 
 
if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())