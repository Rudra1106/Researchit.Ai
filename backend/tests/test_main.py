"""
test_main.py

Testing FastAPI endpoints with TestClient.

TestClient lets us call endpoints as if they were real HTTP requests
but entirely in-process — no server, no network, no Ollama, no PDF needed.

We mock the heavy dependencies (process_pdf, chat) so tests run
in milliseconds and focus purely on HTTP behaviour:
  - correct status codes
  - correct response shapes
  - correct error messages for bad input
  - correct flow: upload → session → chat

Run with: pytest backend/tests/test_main.py -v
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app, vector_store, session_manager

client = TestClient(app)


# ── Shared fake data ───────────────────────────────────────────────────────────

FAKE_CHUNKS = [
    {
        "chunk_id": 0, "section": "Abstract", "page": 1,
        "text": "We propose a new architecture called the Transformer.",
        "word_count": 10, "has_math": False, "has_code": False,
    },
    {
        "chunk_id": 1, "section": "Methods", "page": 4,
        "text": r"Attention is computed as softmax(QK^T/sqrt(d_k))V",
        "word_count": 10, "has_math": True, "has_code": False,
    },
]

FAKE_PDF_BYTES = b"%PDF-1.4 fake pdf content for testing"


# ── Helpers ────────────────────────────────────────────────────────────────────

def upload_fake_pdf():
    """
    Helper: simulate a PDF upload and return the session_id.
    Used in tests that need a valid session before they can test /chat.
    """
    with patch("main.process_pdf", return_value=FAKE_CHUNKS):
        response = client.post(
            "/upload",
            files={"file": ("paper.pdf", io.BytesIO(FAKE_PDF_BYTES), "application/pdf")},
        )
    assert response.status_code == 200
    return response.json()["session_id"]


# ── Health endpoint ────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_shape(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "ollama_running" in data
        assert "chunks_stored" in data
        assert data["status"] == "ok"


# ── Upload endpoint ────────────────────────────────────────────────────────────

class TestUpload:

    def test_upload_valid_pdf_returns_200(self):
        with patch("main.process_pdf", return_value=FAKE_CHUNKS):
            response = client.post(
                "/upload",
                files={"file": ("paper.pdf", io.BytesIO(FAKE_PDF_BYTES), "application/pdf")},
            )
        assert response.status_code == 200

    def test_upload_returns_session_id(self):
        with patch("main.process_pdf", return_value=FAKE_CHUNKS):
            response = client.post(
                "/upload",
                files={"file": ("paper.pdf", io.BytesIO(FAKE_PDF_BYTES), "application/pdf")},
            )
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_upload_returns_chunk_count(self):
        with patch("main.process_pdf", return_value=FAKE_CHUNKS):
            response = client.post(
                "/upload",
                files={"file": ("paper.pdf", io.BytesIO(FAKE_PDF_BYTES), "application/pdf")},
            )
        data = response.json()
        assert data["chunk_count"] == len(FAKE_CHUNKS)

    def test_upload_returns_sections(self):
        with patch("main.process_pdf", return_value=FAKE_CHUNKS):
            response = client.post(
                "/upload",
                files={"file": ("paper.pdf", io.BytesIO(FAKE_PDF_BYTES), "application/pdf")},
            )
        data = response.json()
        assert "sections" in data
        assert isinstance(data["sections"], list)

    def test_upload_non_pdf_rejected(self):
        """Uploading a .txt file should return 400."""
        response = client.post(
            "/upload",
            files={"file": ("notes.txt", io.BytesIO(b"some text"), "text/plain")},
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_upload_empty_pdf_returns_422(self):
        """A PDF that yields no chunks should return 422 with a helpful message."""
        with patch("main.process_pdf", return_value=[]):
            response = client.post(
                "/upload",
                files={"file": ("empty.pdf", io.BytesIO(FAKE_PDF_BYTES), "application/pdf")},
            )
        assert response.status_code == 422
        assert "extract" in response.json()["detail"].lower()


# ── Chat endpoint ──────────────────────────────────────────────────────────────

class TestChat:

    def test_chat_valid_request_returns_200(self):
        session_id = upload_fake_pdf()
        with patch("main.chat", return_value="Attention is a mechanism..."):
            response = client.post("/chat", json={
                "session_id": session_id,
                "question": "What is attention?",
            })
        assert response.status_code == 200

    def test_chat_response_has_answer_and_sources(self):
        session_id = upload_fake_pdf()
        with patch("main.chat", return_value="Here is the answer."):
            response = client.post("/chat", json={
                "session_id": session_id,
                "question": "Explain self-attention.",
            })
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["answer"] == "Here is the answer."

    def test_chat_invalid_session_returns_404(self):
        """Using a made-up session_id should return 404."""
        response = client.post("/chat", json={
            "session_id": "does-not-exist",
            "question":   "What is attention?",
        })
        assert response.status_code == 404

    def test_chat_empty_question_returns_400(self):
        session_id = upload_fake_pdf()
        response = client.post("/chat", json={
            "session_id": session_id,
            "question":   "   ",   # whitespace only
        })
        assert response.status_code == 400

    def test_chat_question_too_long_returns_400(self):
        session_id = upload_fake_pdf()
        response = client.post("/chat", json={
            "session_id": session_id,
            "question":   "a" * 1001,
        })
        assert response.status_code == 400

    def test_chat_ollama_down_returns_503(self):
        session_id = upload_fake_pdf()
        with patch("main.chat", side_effect=ConnectionError("Ollama not running")):
            response = client.post("/chat", json={
                "session_id": session_id,
                "question":   "What is attention?",
            })
        assert response.status_code == 503

    def test_chat_timeout_returns_504(self):
        session_id = upload_fake_pdf()
        with patch("main.chat", side_effect=TimeoutError()):
            response = client.post("/chat", json={
                "session_id": session_id,
                "question":   "What is attention?",
            })
        assert response.status_code == 504

    def test_sources_contain_section_and_page(self):
        """Sources returned with the answer should have section, page, and preview."""
        session_id = upload_fake_pdf()
        with patch("main.chat", return_value="Answer."):
            response = client.post("/chat", json={
                "session_id": session_id,
                "question":   "explain the model",
            })
        sources = response.json()["sources"]
        if sources:   # only check if chunks were actually retrieved
            assert "section" in sources[0]
            assert "page"    in sources[0]
            assert "preview" in sources[0]
            assert "score"   in sources[0]


# ── Session endpoint ───────────────────────────────────────────────────────────

class TestSession:

    def test_get_session_returns_200(self):
        session_id = upload_fake_pdf()
        response = client.get(f"/session/{session_id}")
        assert response.status_code == 200

    def test_get_session_returns_correct_fields(self):
        session_id = upload_fake_pdf()
        response = client.get(f"/session/{session_id}")
        data = response.json()
        assert data["session_id"] == session_id
        assert "paper_filename" in data
        assert "turn_count" in data
        assert "history_length" in data
        assert "created_at" in data

    def test_get_missing_session_returns_404(self):
        response = client.get("/session/fake-id-123")
        assert response.status_code == 404

    def test_turn_count_increments_after_chat(self):
        """After one chat turn, turn_count should be 1."""
        session_id = upload_fake_pdf()
        with patch("main.chat", return_value="Answer."):
            client.post("/chat", json={
                "session_id": session_id,
                "question":   "First question",
            })
        response = client.get(f"/session/{session_id}")
        assert response.json()["turn_count"] == 1

    def test_delete_session_returns_200(self):
        session_id = upload_fake_pdf()
        response = client.delete(f"/session/{session_id}")
        assert response.status_code == 200

    def test_delete_session_then_chat_returns_404(self):
        """After deleting a session, /chat should return 404."""
        session_id = upload_fake_pdf()
        client.delete(f"/session/{session_id}")
        response = client.post("/chat", json={
            "session_id": session_id,
            "question":   "anything",
        })
        assert response.status_code == 404

    def test_history_length_grows_with_turns(self):
        """history_length should reflect number of conversation turns."""
        session_id = upload_fake_pdf()
        with patch("main.chat", return_value="Answer."):
            for i in range(3):
                client.post("/chat", json={
                    "session_id": session_id,
                    "question":   f"Question {i}",
                })
        response = client.get(f"/session/{session_id}")
        assert response.json()["history_length"] == 3