"""
session_store.py

Responsibility: track conversation state per session.

A session holds:
  - conversation history (so the LLM remembers context)
  - which paper is loaded
  - when the session started

In phase 1, sessions live in a plain Python dict (in memory).
If the server restarts, sessions are lost. That is fine for now.
In a later phase we can persist this to SQLite or Redis.
"""

import uuid
from datetime import datetime



# Maximum conversation turns to keep per session.
# Beyond this we drop the oldest turn to stay within the LLM context window.
MAX_HISTORY_TURNS = 6


class SessionStore:
    """
    Manages all active sessions.

    Usage:
        store = SessionStore()
        session_id = store.create_session()
        store.add_turn(session_id, "What is attention?", "Attention is...")
        history = store.get_history(session_id)
    """

    def __init__(self):
        # { session_id: session_dict }
        self._sessions = {}

    def create_session(self, paper_filename=None):
        """
        Create a new session and return its ID.
        The session ID is a UUID — guaranteed unique, impossible to guess.
        """
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "history":        [],
            "paper_filename": paper_filename,
            "created_at":     datetime.utcnow().isoformat(),
            "turn_count":     0,
        }
        return session_id

    def session_exists(self, session_id):
        return session_id in self._sessions

    def get_session(self, session_id):
        """Return the full session dict, or None if not found."""
        return self._sessions.get(session_id)

    def get_history(self, session_id):
        """
        Return conversation history as a list of (question, answer) tuples.
        Returns [] if session doesn't exist.
        """
        session = self._sessions.get(session_id)
        if not session:
            return []
        return session["history"]

    def add_turn(self, session_id, question, answer):
        """
        Append a (question, answer) pair to the session history.
        Trims oldest turn if we exceed MAX_HISTORY_TURNS.
        """
        if not self.session_exists(session_id):
            return

        session = self._sessions[session_id]
        session["history"].append((question, answer))
        session["turn_count"] += 1

        # Trim: keep only the most recent turns
        if len(session["history"]) > MAX_HISTORY_TURNS:
            session["history"].pop(0)

    def set_paper(self, session_id, filename):
        """Record which paper is loaded in this session."""
        if self.session_exists(session_id):
            self._sessions[session_id]["paper_filename"] = filename

    def clear_session(self, session_id):
        """Reset history but keep the session alive."""
        if self.session_exists(session_id):
            self._sessions[session_id]["history"] = []
            self._sessions[session_id]["paper_filename"] = None

    def delete_session(self, session_id):
        """Remove the session entirely."""
        self._sessions.pop(session_id, None)

    def list_sessions(self):
        """Return all active session IDs — useful for debugging."""
        return list(self._sessions.keys())