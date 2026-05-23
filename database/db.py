import sqlite3


conn = sqlite3.connect("database/followup_questions.db")
cursor = conn.cursor()

def create_table():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT,
        question TEXT
        status TEXT DEFAULT 'unanswered'
    )
    """)
    conn.commit()

def add_question(conversation_id, question):
    conn = sqlite3.connect("database/followup_questions.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO questions (conversation_id, question) VALUES (?, ?)", (conversation_id, question))
    conn.commit()
    conn.close()

def get_questions(conversation_id):
    conn = sqlite3.connect("database/followup_questions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT question FROM questions WHERE conversation_id = ? AND status = 'unanswered'", (conversation_id,))
    questions = cursor.fetchall()
    conn.close()
    return [q[0] for q in questions]

def mark_question_answered(conversation_id, question):
    conn = sqlite3.connect("database/followup_questions.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE questions SET status = 'answered' WHERE conversation_id = ? AND question = ?", (conversation_id, question))
    conn.commit()
    conn.close()
