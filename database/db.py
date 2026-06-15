import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "followup_questions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

def create_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            question TEXT,
            status TEXT DEFAULT 'waiting',
            answer TEXT,
            topic_cluster TEXT DEFAULT 'General'
        )
    """)
    conn.commit()
    conn.close()

def add_question(conversation_id, question, topic_cluster="General"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO questions (conversation_id, question, topic_cluster) VALUES (?, ?, ?)",
        (conversation_id, question, topic_cluster)
    )
    conn.commit()
    print(f"[db] inserted: conv={conversation_id} q={question[:60]} cluster={topic_cluster}")
    conn.close()

def get_questions(conversation_id: str = None):
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT conversation_id, question, topic_cluster, status FROM questions WHERE status = 'waiting'"
    params = []
    if conversation_id:
        query += " AND conversation_id = ?"
        params.append(conversation_id)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [{"conversation_id": r[0], "question": r[1], "topic_cluster": r[2], "status": r[3]} for r in rows]

def get_conversation_context(conversation_id, question):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT context FROM questions WHERE conversation_id = ? AND question = ?",
        (conversation_id, question)
    )
    rows = cursor.fetchall()
    conn.close()
    return "||".join([r[0] for r in rows if r[0]])

def mark_question_answered(conversation_id, question, answer):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE questions SET status = 'answered', answer = ? WHERE conversation_id = ? AND question = ?",
        (answer, conversation_id, question)
    )
    conn.commit()
    print(f"[db] mark_answered: rows_affected={cursor.rowcount} conv={conversation_id} q={question[:60]}")
    conn.close()

def get_answered_questions(conversation_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT question, answer FROM questions WHERE conversation_id = ? AND status = 'answered'",
        (conversation_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"question": r[0], "answer": r[1]} for r in rows]

create_table()