"""
src/api.py
==========
Bonus Task B: Episodic memory backend (FastAPI + SQLite)

REST API for managing conversations and episodic memory.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Episodic Memory Backend API")

DB_PATH = Path(__file__).parent.parent / "episodic_memory.db"

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    # Ενεργοποίηση των Foreign Keys στο SQLite (για το DELETE CASCADE)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

# ── 1. Database Schema ────────────────────────────────────────────────────────
def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                title TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES Conversations(id) ON DELETE CASCADE
            )
        """)
init_db()

# ── 2. Pydantic Models ────────────────────────────────────────────────────────
class MessagePair(BaseModel):
    conversation_id: Optional[int] = None
    user_content: str
    ai_content: str
    title: Optional[str] = "New Chat Thread"

# ── 3. REST API Endpoints ─────────────────────────────────────────────────────

@app.get("/conversations")
def list_conversations():
    """List all threads."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM Conversations ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]

@app.get("/conversation/{conversation_id}")
def get_conversation(conversation_id: int):
    """Retrieve a specific thread's history."""
    with get_db() as conn:
        # Έλεγχος αν υπάρχει η συζήτηση
        conv = conn.execute("SELECT id FROM Conversations WHERE id = ?", (conversation_id,)).fetchone()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        rows = conn.execute("SELECT * FROM Messages WHERE conversation_id = ? ORDER BY id ASC", (conversation_id,)).fetchall()
        return [dict(row) for row in rows]

@app.post("/message")
def store_message_pair(payload: MessagePair):
    """Store a new user/AI message pair."""
    with get_db() as conn:
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        
        conv_id = payload.conversation_id
        
        # Αν δεν δόθηκε ID, φτιάχνουμε νέα συζήτηση (thread)
        if not conv_id:
            cursor.execute("INSERT INTO Conversations (created_at, title) VALUES (?, ?)", (now, payload.title))
            conv_id = cursor.lastrowid
        
        # Αποθήκευση του User Message
        cursor.execute("INSERT INTO Messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)", 
                       (conv_id, "user", payload.user_content, now))
        
        # Αποθήκευση του AI Message
        cursor.execute("INSERT INTO Messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)", 
                       (conv_id, "assistant", payload.ai_content, now))
        
        conn.commit()
        return {"status": "success", "conversation_id": conv_id}

@app.delete("/conversation/{conversation_id}")
def delete_conversation(conversation_id: int):
    """Delete a thread and its messages."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"status": "deleted", "conversation_id": conversation_id}