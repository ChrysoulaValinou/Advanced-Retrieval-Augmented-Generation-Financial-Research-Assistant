"""
src/memory.py
=============
Task 3 – Memory Layers

Implements two distinct memory layers using Google Gemini for summarisation:

  3.1  ShortTermMemory  – volatile conversational buffer (in-memory)
       • Stores the last N message turns as {"role": ..., "content": ...} dicts
       • Prepended to every LLM prompt to maintain within-session coherence
       • Cleared automatically when the session ends (process exits)

  3.2  LongTermMemory   – persistent cross-session memory (SQLite)
       • At session end: the Gemini LLM summarises the conversation → saved to DB
       • At session start: the saved summary is injected into the system prompt
       • Survives restarts; grows richer with each session
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# LangChain Google GenAI imports replacing OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "memory.db"


# ══════════════════════════════════════════════════════════════════════════════════
# 3.1  Short-term (working) memory
# ══════════════════════════════════════════════════════════════════════════════════

class ShortTermMemory:
    """
    Volatile conversational buffer — lives only for the current session.

    Stores the last `max_turns` exchanges as a list of
    {"role": "user"|"assistant", "content": str} dicts.
    """

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns: int        = max_turns
        self._buffer:  list[dict]  = []

    # ── Public API ────────────────────────────────────────────────────────────────

    def add_user(self, content: str) -> None:
        """Append a user message and trim the buffer if needed."""
        self._append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        """Append an assistant message and trim the buffer if needed."""
        self._append({"role": "assistant", "content": content})

    def get_history(self) -> list[dict]:
        """
        Return the current buffer as a list of message dicts.
        """
        return list(self._buffer)          # return a copy

    def clear(self) -> None:
        """Wipe the buffer (called at session end before saving summary)."""
        self._buffer.clear()
        log.debug("Short-term memory cleared.")

    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ShortTermMemory(turns={len(self._buffer)}, max={self.max_turns})"

    # ── Internal ──────────────────────────────────────────────────────────────────

    def _append(self, message: dict) -> None:
        self._buffer.append(message)
        # Trim to max_turns — drop from the front (oldest messages first)
        if len(self._buffer) > self.max_turns:
            dropped = self._buffer.pop(0)
            log.debug("Short-term buffer full — dropped oldest message: role=%s",
                      dropped["role"])

    def format_for_prompt(self) -> str:
        """
        Format the buffer as a readable block for debugging / logging.
        """
        if not self._buffer:
            return "(no conversation history)"
        lines = []
        for msg in self._buffer:
            role    = msg["role"].upper()
            content = msg["content"][:120].replace("\n", " ")
            lines.append(f"  [{role}]: {content}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════════
# 3.2  Long-term (persistent) memory
# ══════════════════════════════════════════════════════════════════════════════════

class LongTermMemory:
    """
    Persistent cross-session memory backed by a SQLite database.
    """

    # Prompt sent to the LLM to generate the summary
    _SUMMARY_PROMPT = """\
You are a memory assistant for a financial research RAG chatbot.

Below is a conversation between a user and the assistant.
Write a concise summary (5-8 sentences maximum) capturing:
  1. The main topics and questions the user asked about
  2. Any preferences or constraints the user expressed (e.g. "I prefer bullet points", "I work in finance")
  3. Any unresolved questions or topics the user seemed interested in continuing
  4. The domain focus (e.g. TechVision Corp, macroeconomic data, competitor analysis)

Do NOT reproduce the conversation verbatim.
Do NOT include greetings or meta-commentary.
Write in third person ("The user asked about…").

--- CONVERSATION ---
{conversation}
--- END ---

Summary:"""

    def __init__(self, db_path: Path = DB_PATH, model: str = "gemini-2.5-flash") -> None:
        self._db_path  = db_path
        # Χρησιμοποιούμε το μοντέλο της Google (temperature=0 για πιο σταθερές περιλήψεις)
        self._llm      = ChatGoogleGenerativeAI(model=model, temperature=0)
        self._init_db()

    # ── DB initialisation ─────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create the sessions table if it does not exist yet."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at    TEXT    NOT NULL,
                    ended_at      TEXT,
                    summary       TEXT,
                    message_count INTEGER DEFAULT 0
                )
            """)
        log.info("LongTermMemory DB ready → %s", self._db_path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    # ── Session start: load context ───────────────────────────────────────────────

    def load_context(self, n_sessions: int = 3) -> str:
        """
        Load summaries from the last `n_sessions` sessions and return
        a combined string to inject into the system prompt.
        """
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT id, started_at, summary
                FROM   sessions
                WHERE  summary IS NOT NULL
                ORDER  BY id DESC
                LIMIT  ?
            """, (n_sessions,)).fetchall()

        if not rows:
            log.info("No previous session summaries found.")
            return ""

        parts = []
        for session_id, started_at, summary in reversed(rows):  # chronological order
            date_str = started_at[:10]   # just the date portion
            parts.append(f"[Session {session_id} — {date_str}]\n{summary}")

        context = "\n\n".join(parts)
        log.info("Loaded %d previous session summary/summaries from DB.", len(rows))
        return context

    # ── Session end: summarise and save ──────────────────────────────────────────

    def save_session_summary(
        self,
        history:       list[dict],
        started_at:    datetime,
    ) -> Optional[str]:
        """
        Ask the Gemini LLM to summarise `history`, then persist to SQLite.
        """
        if not history:
            log.info("No conversation to summarise — skipping long-term save.")
            return None

        # Format conversation for the summary prompt
        conversation_text = self._format_history(history)

        # Ask LLM to summarise
        log.info("Generating session summary via Gemini LLM …")
        try:
            prompt_text = self._SUMMARY_PROMPT.format(conversation=conversation_text)
            messages = [HumanMessage(content=prompt_text)]
            
            response = self._llm.invoke(messages)
            summary = response.content.strip()
        except Exception as exc:
            log.error("LLM summarisation failed: %s", exc)
            # Fallback: save a truncated raw snippet instead of nothing
            summary = f"[Summary generation failed] Topics: {conversation_text[:200]}"

        # Persist to DB
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO sessions (started_at, ended_at, summary, message_count)
                VALUES (?, ?, ?, ?)
            """, (
                started_at.isoformat(),
                now,
                summary,
                len(history),
            ))

        log.info("Session summary saved to DB (%d chars).", len(summary))
        return summary

    # ── Utilities ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        """Convert list of message dicts to a readable conversation string."""
        lines = []
        for msg in history:
            role    = msg["role"].capitalize()
            content = msg["content"]
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    def get_all_sessions(self) -> list[dict]:
        """Return all stored sessions as a list of dicts (for debugging/inspection)."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT id, started_at, ended_at, message_count,
                       SUBSTR(summary, 1, 120) AS summary_preview
                FROM   sessions
                ORDER  BY id DESC
            """).fetchall()
        return [
            {
                "id":             r[0],
                "started_at":     r[1],
                "ended_at":       r[2],
                "message_count":  r[3],
                "summary_preview": r[4],
            }
            for r in rows
        ]

    def delete_session(self, session_id: int) -> bool:
        """Delete a specific session by id. Returns True if a row was deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
        deleted = cursor.rowcount > 0
        if deleted:
            log.info("Deleted session %d from long-term memory.", session_id)
        return deleted

    def __repr__(self) -> str:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return f"LongTermMemory(db={self._db_path.name}, sessions={count})"


# ══════════════════════════════════════════════════════════════════════════════════
# MemoryManager — convenience wrapper used by main.py
# ══════════════════════════════════════════════════════════════════════════════════

class MemoryManager:
    """
    Thin facade that wires ShortTermMemory and LongTermMemory together.
    """

    def __init__(
        self,
        max_short_term_turns: int  = 6,
        db_path:              Path = DB_PATH,
        llm_model:            str  = "gemini-1.5-flash",
    ) -> None:
        self.short = ShortTermMemory(max_turns=max_short_term_turns)
        self.long  = LongTermMemory(db_path=db_path, model=llm_model)
        self._session_start = datetime.now(timezone.utc)
        log.info("MemoryManager initialised. Session start: %s",
                 self._session_start.isoformat())

    def build_system_prompt(self, base_prompt: str) -> str:
        """
        Inject long-term memory context into the base system prompt.
        If no previous sessions exist, returns base_prompt unchanged.
        """
        lt_context = self.long.load_context(n_sessions=3)
        if not lt_context:
            return base_prompt

        return (
            f"{base_prompt}\n\n"
            f"## Memory from Previous Sessions\n"
            f"The following is a summary of what you know about this user "
            f"from past conversations. Use it to personalise your responses.\n\n"
            f"{lt_context}"
        )

    def add_user_turn(self, content: str) -> None:
        self.short.add_user(content)

    def add_assistant_turn(self, content: str) -> None:
        self.short.add_assistant(content)

    def get_short_term_history(self) -> list[dict]:
        """Return the short-term buffer — pass directly to the LLM messages array."""
        return self.short.get_history()

    def end_session(self) -> None:
        """
        Called when the user quits.
        1. Captures the short-term buffer before clearing it.
        2. Asks the LLM to summarise the session.
        3. Persists the summary to SQLite.
        4. Clears the short-term buffer.
        """
        history = self.short.get_history()    # snapshot before clear
        self.long.save_session_summary(
            history=history,
            started_at=self._session_start,
        )
        self.short.clear()
        log.info("Session ended. Memory saved.")

    def __repr__(self) -> str:
        return (f"MemoryManager(\n"
                f"  short={self.short!r}\n"
                f"  long={self.long!r}\n"
                f"  session_start={self._session_start.isoformat()}\n"
                f")")