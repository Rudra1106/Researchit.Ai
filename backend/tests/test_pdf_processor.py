"""
test_pdf_processor.py

We test each function in isolation so we know exactly what broke if something fails.
This is called "unit testing" — testing one unit of behavior at a time.

Run with:  pytest backend/tests/test_pdf_processor.py -v
The -v flag means "verbose" — shows each test name and pass/fail.
"""

import pytest
import sys
import os

# Add the backend directory to Python's path so we can import pdf_processor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from 
from pdf_processor import (
    _clean_text,
    _detect_math,
    _detect_code,
    _is_footnote,
    _is_figure_caption,
    chunk_blocks,
    make_chunk,
)


# ── Tests for _clean_text ──────────────────────────────────────────────────────

class TestCleanText:
    """Group related tests in a class — pytest will find and run them all."""
    
    def test_removes_hyphenated_linebreaks(self):
        """'atten-\ntion' should become 'attention'"""
        result = _clean_text("atten-\ntion mechanism")
        assert "attention" in result
    
    def test_collapses_multiple_spaces(self):
        """Multiple spaces should become one."""
        result = _clean_text("hello    world")
        assert result == "hello world"
    
    def test_removes_lone_page_numbers(self):
        """A line that is just a number (like '4') should be removed."""
        result = _clean_text("some text\n4\nmore text")
        assert "4" not in result.split()  # '4' shouldn't appear as a standalone word
    
    def test_preserves_real_numbers_in_sentences(self):
        """Numbers inside sentences should survive."""
        result = _clean_text("The model has 512 dimensions")
        assert "512" in result
    
    def test_strips_whitespace(self):
        """Leading and trailing whitespace should be removed."""
        result = _clean_text("   hello   ")
        assert result == "hello"


# ── Tests for _detect_math ────────────────────────────────────────────────────

class TestDetectMath:
    
    def test_detects_frac(self):
        assert _detect_math(r"The formula is \frac{a}{b}") is True
    
    def test_detects_inline_dollar(self):
        assert _detect_math("where $x$ is the input") is True
    
    def test_detects_display_math(self):
        assert _detect_math("$$E = mc^2$$") is True
    
    def test_detects_superscript(self):
        assert _detect_math("x^{T}y is the dot product") is True
    
    def test_no_math_in_plain_text(self):
        assert _detect_math("This is a plain sentence with no math.") is False
    
    def test_no_false_positive_on_dollar_sign(self):
        """A lone $ (like a price) shouldn't trigger math detection."""
        # Our pattern requires $...$ with content inside
        assert _detect_math("costs $50") is False


# ── Tests for _detect_code ────────────────────────────────────────────────────

class TestDetectCode:
    
    def test_detects_python_function(self):
        code = "def forward(self, x):\n    return self.linear(x)"
        assert _detect_code(code) is True
    
    def test_detects_import(self):
        assert _detect_code("import torch\nimport numpy as np") is True
    
    def test_no_code_in_plain_text(self):
        assert _detect_code("We propose a new attention mechanism.") is False
    
    def test_detects_indented_block(self):
        code = "  x = 1\n  y = 2\n  z = x + y"
        assert _detect_code(code) is True


# ── Tests for _is_footnote ─────────────────────────────────────────────────────

class TestIsFootnote:
    
    def test_block_at_bottom_is_footnote(self):
        """A block at 90% of page height should be a footnote."""
        block = {"bbox": (0, 900, 100, 910)}  # y0=900 on a 1000-height page
        assert _is_footnote(block, page_height=1000) is True
    
    def test_block_in_middle_is_not_footnote(self):
        """A block at 50% of page height is body text."""
        block = {"bbox": (0, 500, 100, 520)}
        assert _is_footnote(block, page_height=1000) is False
    
    def test_block_just_above_threshold(self):
        """A block at exactly 88% should NOT be a footnote (threshold is > 0.88)."""
        block = {"bbox": (0, 880, 100, 890)}
        assert _is_footnote(block, page_height=1000) is False


# ── Tests for _is_figure_caption ──────────────────────────────────────────────

class TestIsFigureCaption:
    
    def test_figure_caption(self):
        assert _is_figure_caption("Figure 3: Architecture of our model.") is True
    
    def test_fig_abbreviation(self):
        assert _is_figure_caption("Fig. 1. Results on ImageNet.") is True
    
    def test_table_caption(self):
        assert _is_figure_caption("Table 2: Comparison of methods.") is True
    
    def test_normal_sentence_is_not_caption(self):
        assert _is_figure_caption("We show in Figure 3 that...") is False
    
    def test_case_insensitive(self):
        assert _is_figure_caption("figure 1: something") is True


# ── Tests for chunk_blocks ─────────────────────────────────────────────────────

class TestChunkBlocks:
    """
    We can test chunk_blocks without a real PDF by creating fake block lists.
    This is the right way to test — we isolate the chunking logic
    from the PDF parsing logic.
    """
    
    def _make_block(self, block_type, text, page=1):
        """Helper to create a fake block dict."""
        return {"type": block_type, "text": text, "page": page, "bbox": (0, 0, 100, 20)}
    
    def test_heading_creates_section(self):
        """A heading block should set the section name for following chunks."""
        blocks = [
            self._make_block("heading", "Introduction"),
            self._make_block("body", "This paper proposes " * 20),  # enough words
        ]
        chunks = chunk_blocks(blocks)
        assert len(chunks) >= 1
        assert chunks[0]["section"] == "Introduction"
    
    def test_body_text_accumulates(self):
        """Multiple body blocks in the same section should merge into one chunk."""
        blocks = [
            self._make_block("heading", "Methods"),
            self._make_block("body", "First sentence. " * 10),
            self._make_block("body", "Second sentence. " * 10),
        ]
        chunks = chunk_blocks(blocks)
        # Both body blocks should be in one chunk (they're under MAX_CHUNK_WORDS)
        assert len(chunks) == 1
        assert "First sentence" in chunks[0]["text"]
        assert "Second sentence" in chunks[0]["text"]
    
    def test_chunk_splits_at_word_limit(self):
        """A very long body block should produce multiple chunks."""
        # 400 words exceeds our 300-word limit
        long_text = "word " * 400
        blocks = [
            self._make_block("heading", "Results"),
            self._make_block("body", long_text),
        ]
        chunks = chunk_blocks(blocks)
        assert len(chunks) >= 2
    
    def test_caption_gets_own_chunk(self):
        """Figure captions should become their own chunk."""
        blocks = [
            self._make_block("heading", "Results"),
            self._make_block("body", "Some result text. " * 15),
            self._make_block("caption", "Figure 1: Our model architecture."),
            self._make_block("body", "More analysis. " * 15),
        ]
        chunks = chunk_blocks(blocks)
        caption_chunks = [c for c in chunks if "Figure 1" in c["text"]]
        assert len(caption_chunks) == 1
    
    def test_chunk_has_correct_metadata_keys(self):
        """Every chunk must have all required metadata fields."""
        blocks = [
            self._make_block("heading", "Abstract"),
            self._make_block("body", "This is the abstract. " * 15),
        ]
        chunks = chunk_blocks(blocks)
        assert len(chunks) > 0
        required_keys = {"chunk_id", "section", "page", "text", "word_count", "has_math", "has_code"}
        for chunk in chunks:
            assert required_keys.issubset(chunk.keys()), f"Missing keys: {required_keys - chunk.keys()}"
    
    def test_math_detection_in_chunk(self):
        """A chunk containing LaTeX should have has_math=True."""
        blocks = [
            self._make_block("heading", "Methods"),
            self._make_block("body", r"We compute \frac{Q K^T}{\sqrt{d_k}} " * 20),
        ]
        chunks = chunk_blocks(blocks)
        assert any(c["has_math"] for c in chunks)
    
    def test_short_text_discarded(self):
        """Blocks with fewer than 10 words should not become chunks."""
        blocks = [
            self._make_block("body", "Too short."),
        ]
        chunks = chunk_blocks(blocks)
        assert len(chunks) == 0
    
    def test_section_carries_across_chunks(self):
        """When a long section splits into multiple chunks, they all share the section name."""
        blocks = [
            self._make_block("heading", "Related Work"),
            self._make_block("body", "word " * 400),  # will split into 2+ chunks
        ]
        chunks = chunk_blocks(blocks)
        assert all(c["section"] == "Related Work" for c in chunks)