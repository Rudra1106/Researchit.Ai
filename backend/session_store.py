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
            "learner_profile": {
                "level":               "beginner",
                "known_concepts":      [],
                "preferred_style":     "analogy",
                "taught_this_session": [],
                "profiled":            False,
            },
            "reading_path":   [],
            "mentor_mode": {
                "active":                  False,
                "topic":                   "",
                "tagline":                 "",
                "steps":                   [],
                "current_step":            0,      # 0-indexed
                "step_status":             [],     # "pending"|"in_progress"|"passed"|"skipped"
                "awaiting_check_answer":   False,
                "check_question":          "",
            },
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

    # ── Learner profile ─────────────────────────────────────────────────────────

    def get_profile(self, session_id: str) -> dict:
        """Return the learner profile for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return {}
        return session.get("learner_profile", {})

    def update_profile(self, session_id: str, profile: dict):
        """Replace the learner profile with a freshly inferred one."""
        if self.session_exists(session_id):
            # Preserve taught_this_session across profile updates
            existing = self._sessions[session_id].get("learner_profile", {})
            profile.setdefault("taught_this_session", existing.get("taught_this_session", []))
            self._sessions[session_id]["learner_profile"] = profile

    def mark_concept_taught(self, session_id: str, concept: str):
        """Record that a concept was taught this session."""
        if not self.session_exists(session_id):
            return
        profile = self._sessions[session_id].setdefault("learner_profile", {})
        taught  = profile.setdefault("taught_this_session", [])
        if concept not in taught:
            taught.append(concept)

    def set_reading_path(self, session_id: str, path: list):
        """Store the adaptive reading path generated at upload time."""
        if self.session_exists(session_id):
            self._sessions[session_id]["reading_path"] = path

    def get_reading_path(self, session_id: str) -> list:
        session = self._sessions.get(session_id)
        return session.get("reading_path", []) if session else []

    # ── Mentor Mode ─────────────────────────────────────────────────────────────────

    def get_mentor_state(self, session_id: str) -> dict:
        session = self._sessions.get(session_id)
        if not session:
            return {"active": False}
        return session.get("mentor_mode", {"active": False})

    def activate_mentor_mode(self, session_id: str, curriculum: dict):
        """Enter Mentor Mode with the generated curriculum."""
        if not self.session_exists(session_id):
            return
        steps  = curriculum.get("steps", [])
        mentor = self._sessions[session_id]["mentor_mode"]
        mentor["active"]              = True
        mentor["topic"]               = curriculum.get("topic", "")
        mentor["tagline"]             = curriculum.get("tagline", "")
        mentor["steps"]               = steps
        mentor["current_step"]        = 0
        mentor["step_status"]         = ["in_progress"] + ["pending"] * (len(steps) - 1)
        mentor["awaiting_check_answer"] = False
        mentor["check_question"]      = ""

    def advance_step(self, session_id: str) -> bool:
        """
        Advance to the next curriculum step.
        Returns True if there is a next step, False if curriculum is complete.
        """
        if not self.session_exists(session_id):
            return False
        mentor = self._sessions[session_id]["mentor_mode"]
        idx    = mentor["current_step"]
        steps  = mentor["steps"]

        # Mark current step done (if not already passed)
        if idx < len(mentor["step_status"]) and mentor["step_status"][idx] != "passed":
            mentor["step_status"][idx] = "skipped"

        if idx + 1 >= len(steps):
            mentor["active"] = False   # curriculum complete
            return False

        mentor["current_step"] = idx + 1
        mentor["step_status"][idx + 1] = "in_progress"
        mentor["awaiting_check_answer"] = False
        mentor["check_question"]        = ""
        return True

    def mark_step_passed(self, session_id: str):
        """Mark the current step as passed (student answered correctly)."""
        if not self.session_exists(session_id):
            return
        mentor = self._sessions[session_id]["mentor_mode"]
        idx    = mentor["current_step"]
        if idx < len(mentor["step_status"]):
            mentor["step_status"][idx] = "passed"

    def set_awaiting_check(self, session_id: str, check_question: str):
        """Record that the mentor asked a Socratic check question."""
        if not self.session_exists(session_id):
            return
        mentor = self._sessions[session_id]["mentor_mode"]
        mentor["awaiting_check_answer"] = bool(check_question)
        mentor["check_question"]        = check_question

    def deactivate_mentor_mode(self, session_id: str):
        """Exit Mentor Mode (curriculum complete or user requested)."""
        if self.session_exists(session_id):
            self._sessions[session_id]["mentor_mode"]["active"] = False