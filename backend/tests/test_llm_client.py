"""
test_llm_client.py

Testing strategy for LLM client code:
  We CANNOT test "is the answer good?" in a unit test — that's subjective
  and slow. What we CAN test:
    * Is the prompt structured correctly?
    * Does history get included in messages?
    * Does chunk context appear in the prompt?
    * Do errors (connection refused, timeout) raise the right exceptions?
    * Does the response parser extract the right field?

Key technique: unittest.mock.patch
  We "mock" the requests.post call so tests never actually hit Ollama.
  Instead, we tell the mock "when called, pretend you got this response."
  This makes tests fast (milliseconds) and runnable without Ollama.

Run with: pytest backend/tests/test_llm_client.py -v
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_client import (
    _format_chunks_as_context,
    _build_messages,
    chat,
    check_ollama_running,
    SYSTEM_PROMPT,
    OLLAMA_MODEL,
)


# ── Sample data ────────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {
        "chunk_id": "1",
        "section": "Methods",
        "page": 4,
        "text": "Attention is computed as softmax(QK^T / sqrt(d_k)) * V",
        "score": 0.91,
        "has_math": True,
        "has_code": False,
    },
    {
        "chunk_id": "2",
        "section": "Introduction",
        "page": 2,
        "text": "The transformer model relies entirely on attention mechanisms.",
        "score": 0.85,
        "has_math": False,
        "has_code": False,
    },
]

SAMPLE_HISTORY = [
    ("What is a transformer?", "A transformer is a neural network architecture..."),
    ("Why is attention useful?",  "Attention lets the model focus on relevant tokens..."),
]


# ── Tests for _format_chunks_as_context ───────────────────────────────────────

class TestFormatChunksAsContext:

    def test_empty_chunks_returns_fallback(self):
        """No chunks should produce a clear 'nothing found' message."""
        result = _format_chunks_as_context([])
        assert "No relevant sections" in result

    def test_chunk_text_appears_in_context(self):
        """The chunk's text must appear in the formatted output."""
        result = _format_chunks_as_context(SAMPLE_CHUNKS)
        assert "softmax(QK^T / sqrt(d_k))" in result
        assert "transformer model relies entirely" in result

    def test_section_label_appears(self):
        """The section name should be visible so the LLM can reference it."""
        result = _format_chunks_as_context(SAMPLE_CHUNKS)
        assert "Methods" in result
        assert "Introduction" in result

    def test_page_number_appears(self):
        """Page numbers help users find the original text."""
        result = _format_chunks_as_context(SAMPLE_CHUNKS)
        assert "4" in result   # page 4
        assert "2" in result   # page 2

    def test_multiple_chunks_are_separated(self):
        """Chunks should be visually separated so the LLM doesn't merge them."""
        result = _format_chunks_as_context(SAMPLE_CHUNKS)
        assert "---" in result   # our separator

    def test_single_chunk(self):
        """Single chunk should still work without a separator."""
        result = _format_chunks_as_context([SAMPLE_CHUNKS[0]])
        assert "softmax" in result


# ── Tests for _build_messages ─────────────────────────────────────────────────

class TestBuildMessages:

    def test_first_message_is_system(self):
        """Messages must start with the system prompt."""
        messages = _build_messages("test question", SAMPLE_CHUNKS, [])
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT

    def test_last_message_is_user(self):
        """The current question should always be the last message."""
        messages = _build_messages("What is attention?", SAMPLE_CHUNKS, [])
        assert messages[-1]["role"] == "user"

    def test_question_appears_in_last_message(self):
        """The user's question text must be in the last message."""
        question = "What is scaled dot-product attention?"
        messages = _build_messages(question, SAMPLE_CHUNKS, [])
        assert question in messages[-1]["content"]

    def test_chunks_appear_in_last_message(self):
        """Retrieved chunk text must be in the user message for grounding."""
        messages = _build_messages("any question", SAMPLE_CHUNKS, [])
        last_content = messages[-1]["content"]
        assert "softmax(QK^T / sqrt(d_k))" in last_content

    def test_history_is_replayed(self):
        """
        Past conversation turns must appear between system and current message.
        This is what gives the LLM memory.
        """
        messages = _build_messages("new question", SAMPLE_CHUNKS, SAMPLE_HISTORY)

        # Extract roles in order
        roles = [m["role"] for m in messages]

        # Should be: system, user, assistant, user, assistant, user (current)
        assert roles[0] == "system"
        assert roles[-1] == "user"

        # History content should appear
        all_content = " ".join(m["content"] for m in messages)
        assert "What is a transformer?" in all_content
        assert "A transformer is a neural network" in all_content

    def test_empty_history_works(self):
        """No history should produce system + single user message."""
        messages = _build_messages("question", SAMPLE_CHUNKS, [])
        assert len(messages) == 2   # system + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_history_alternates_user_assistant(self):
        """History must be properly interleaved: user, assistant, user, assistant..."""
        messages = _build_messages("q", [], SAMPLE_HISTORY)
        # Skip system message, check history is interleaved
        history_messages = messages[1:-1]   # exclude system and current question
        for i, msg in enumerate(history_messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == expected_role, (
                f"Message {i} should be '{expected_role}' but is '{msg['role']}'"
            )


# ── Tests for chat() — mocking Ollama ─────────────────────────────────────────

class TestChat:
    """
    We mock requests.post so these tests never need Ollama running.

    How mocking works:
        @patch("llm_client.requests.post")
        def test_something(self, mock_post):
            mock_post.return_value = <a fake response object>
            # now when chat() calls requests.post, it gets our fake response

    This is a fundamental testing skill — you'll use this pattern
    any time your code calls an external service.
    """

    def _make_mock_response(self, content="This is the model's answer."):
        """Helper: create a fake requests.Response object."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": OLLAMA_MODEL,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "done": True,
        }
        return mock_response

    @patch("llm_client.requests.post")
    def test_chat_returns_string(self, mock_post):
        """chat() must return a string."""
        mock_post.return_value = self._make_mock_response()
        result = chat("What is attention?", SAMPLE_CHUNKS)
        assert isinstance(result, str)

    @patch("llm_client.requests.post")
    def test_chat_returns_model_content(self, mock_post):
        """chat() should return exactly what the model said."""
        expected = "Attention is a mechanism that..."
        mock_post.return_value = self._make_mock_response(expected)
        result = chat("explain attention", SAMPLE_CHUNKS)
        assert result == expected

    @patch("llm_client.requests.post")
    def test_chat_calls_correct_url(self, mock_post):
        """Verify we're calling the right Ollama endpoint."""
        mock_post.return_value = self._make_mock_response()
        chat("question", [])
        call_args = mock_post.call_args
        assert "api/chat" in call_args[0][0]

    @patch("llm_client.requests.post")
    def test_chat_sends_correct_model(self, mock_post):
        """The payload must specify our chosen model."""
        mock_post.return_value = self._make_mock_response()
        chat("question", [])
        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == OLLAMA_MODEL

    @patch("llm_client.requests.post")
    def test_chat_default_history_is_empty(self, mock_post):
        """
        Calling chat() without a history argument should not crash.
        The default [] means: no previous conversation.
        """
        mock_post.return_value = self._make_mock_response()
        result = chat("first question ever", SAMPLE_CHUNKS)
        # Just checking it doesn't raise — the result should be a string
        assert isinstance(result, str)

    @patch("llm_client.requests.post")
    def test_chat_with_history(self, mock_post):
        """History should be passed through to the messages."""
        mock_post.return_value = self._make_mock_response()
        chat("follow-up question", SAMPLE_CHUNKS, history=SAMPLE_HISTORY)

        payload = mock_post.call_args[1]["json"]
        all_content = " ".join(m["content"] for m in payload["messages"])
        assert "What is a transformer?" in all_content

    @patch("llm_client.requests.post")
    def test_http_error_raises_runtime_error(self, mock_post):
        """A non-200 response should raise RuntimeError with a useful message."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError, match="500"):
            chat("question", [])

    @patch("llm_client.requests.post")
    def test_connection_error_raises_connection_error(self, mock_post):
        """If Ollama isn't running, we should get a clear ConnectionError."""
        import requests as req
        mock_post.side_effect = req.exceptions.ConnectionError()

        with pytest.raises(ConnectionError, match="localhost:11434"):
            chat("question", [])

    @patch("llm_client.requests.post")
    def test_timeout_raises_timeout_error(self, mock_post):
        """Slow responses should raise TimeoutError."""
        import requests as req
        mock_post.side_effect = req.exceptions.Timeout()

        with pytest.raises(TimeoutError):
            chat("question", [])


# ── Tests for check_ollama_running ────────────────────────────────────────────

class TestCheckOllamaRunning:

    @patch("llm_client.requests.get")
    def test_returns_true_when_ollama_up(self, mock_get):
        """Should return True when Ollama responds with 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        assert check_ollama_running() is True

    @patch("llm_client.requests.get")
    def test_returns_false_when_ollama_down(self, mock_get):
        """Should return False if connection is refused."""
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError()
        assert check_ollama_running() is False