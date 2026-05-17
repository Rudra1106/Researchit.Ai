"""
test_vector_store.py

Testing strategy for a vector store:
  - We can't test "did it find the RIGHT chunk" perfectly, because
    similarity search is probabilistic. But we CAN test:
    * Did it store the right number of chunks?
    * Does search return the expected number of results?
    * Does the result have all the fields we need?
    * Does filtering work correctly?
    * Does a clearly relevant query beat a clearly irrelevant one?

Run with: pytest backend/tests/test_vector_store.py -v
"""

import pytest
import sys
import os
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import VectorStore, EMBEDDING_MODEL_NAME


# ── Test fixtures ──────────────────────────────────────────────────────────────
#
# A pytest "fixture" is a function that sets up something tests need.
# By using fixtures we avoid repeating setup code in every test.

@pytest.fixture(scope="module")
def embedding_model():
    """
    Load the embedding model once for all tests in this file.
    'scope=module' means it runs once, not once per test — model loading is slow.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@pytest.fixture
def store(embedding_model):
    """
    Create a fresh in-memory VectorStore for each test.
    We subclass VectorStore to inject an in-memory ChromaDB client
    instead of the on-disk one — tests should never touch disk.
    """
    # We'll monkey-patch the VectorStore to use EphemeralClient
    # by subclassing it just for tests.
    class InMemoryVectorStore(VectorStore):
        def __init__(self):
            print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            self._model = embedding_model  # reuse already-loaded model
            self._client = chromadb.EphemeralClient()  # in-memory, not disk
            self._collection = self._client.get_or_create_collection(
                name="test_chunks",
                metadata={"hnsw:space": "cosine"},
            )

    return InMemoryVectorStore()


@pytest.fixture
def sample_chunks():
    """
    A small set of fake chunks that simulate what pdf_processor would produce.
    We make them about clearly different topics so we can test that search
    returns the right one.
    """
    return [
        {
            "chunk_id": 0,
            "section": "Abstract",
            "page": 1,
            "text": "The transformer architecture uses self-attention mechanisms to process sequences.",
            "word_count": 13,
            "has_math": False,
            "has_code": False,
        },
        {
            "chunk_id": 1,
            "section": "Methods",
            "page": 4,
            "text": r"Attention scores are computed as softmax(\frac{QK^T}{\sqrt{d_k}})V where Q K V are matrices.",
            "word_count": 17,
            "has_math": True,
            "has_code": False,
        },
        {
            "chunk_id": 2,
            "section": "Methods",
            "page": 5,
            "text": "def scaled_dot_product_attention(Q, K, V): scores = Q @ K.T / sqrt(d_k); return softmax(scores) @ V",
            "word_count": 18,
            "has_math": False,
            "has_code": True,
        },
        {
            "chunk_id": 3,
            "section": "Results",
            "page": 8,
            "text": "Our model achieves 28.4 BLEU on WMT 2014 English-to-German translation task.",
            "word_count": 14,
            "has_math": False,
            "has_code": False,
        },
        {
            "chunk_id": 4,
            "section": "Introduction",
            "page": 2,
            "text": "Recurrent neural networks have been the dominant approach for sequence modelling tasks.",
            "word_count": 14,
            "has_math": False,
            "has_code": False,
        },
    ]


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestAddChunks:

    def test_add_chunks_increases_count(self, store, sample_chunks):
        """After adding 5 chunks, the store should contain 5 items."""
        assert store._collection.count() == 0
        store.add_chunks(sample_chunks)
        assert store._collection.count() == 5

    def test_add_empty_list_does_not_crash(self, store):
        """Passing an empty list should be a no-op, not an error."""
        store.add_chunks([])
        assert store._collection.count() == 0

    def test_upsert_does_not_duplicate(self, store, sample_chunks):
        """Adding the same chunks twice should not create duplicates."""
        store.add_chunks(sample_chunks)
        store.add_chunks(sample_chunks)  # same ids → upsert updates in place
        assert store._collection.count() == 5

    def test_metadata_is_stored_correctly(self, store, sample_chunks):
        """The metadata fields should survive the round-trip into ChromaDB."""
        store.add_chunks(sample_chunks)
        raw = store._collection.get(ids=["1"], include=["metadatas"])
        meta = raw["metadatas"][0]
        assert meta["section"] == "Methods"
        assert meta["page"] == 4
        assert meta["has_math"] is True
        assert meta["has_code"] is False


class TestSearch:

    def test_search_returns_list(self, store, sample_chunks):
        """search() should always return a list."""
        store.add_chunks(sample_chunks)
        results = store.search("attention mechanism")
        assert isinstance(results, list)

    def test_search_returns_correct_number(self, store, sample_chunks):
        """Asking for 3 results should give at most 3 results."""
        store.add_chunks(sample_chunks)
        results = store.search("attention", n_results=3)
        assert len(results) <= 3

    def test_result_has_required_fields(self, store, sample_chunks):
        """Every result must have the fields downstream code depends on."""
        store.add_chunks(sample_chunks)
        results = store.search("attention")
        assert len(results) > 0
        required = {"chunk_id", "text", "section", "page", "has_math", "has_code", "score"}
        for r in results:
            assert required.issubset(r.keys()), f"Missing: {required - r.keys()}"

    def test_score_is_between_0_and_1(self, store, sample_chunks):
        """Cosine similarity scores should be in [0, 1]."""
        store.add_chunks(sample_chunks)
        results = store.search("self attention transformer")
        for r in results:
            assert 0.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"

    def test_relevant_query_scores_higher_than_irrelevant(self, store, sample_chunks):
        """
        This is the key semantic test.
        Searching 'softmax attention formula' should rank the math chunk
        higher than the results chunk about BLEU scores.
        """
        store.add_chunks(sample_chunks)
        results = store.search("softmax attention formula", n_results=5)

        # Find scores for our two "control" chunks
        math_chunk_score = next(
            (r["score"] for r in results if r["chunk_id"] == "1"), 0
        )
        results_chunk_score = next(
            (r["score"] for r in results if r["chunk_id"] == "3"), 0
        )
        assert math_chunk_score > results_chunk_score, (
            f"Expected math chunk (score={math_chunk_score:.3f}) to rank higher "
            f"than results chunk (score={results_chunk_score:.3f})"
        )

    def test_search_empty_store_returns_empty_list(self, store):
        """Searching an empty store should return [] not raise an error."""
        results = store.search("anything")
        assert results == []

    def test_search_returns_text(self, store, sample_chunks):
        """Results must include the actual text, not just metadata."""
        store.add_chunks(sample_chunks)
        results = store.search("translation BLEU score")
        assert all(len(r["text"]) > 0 for r in results)


class TestSearchWithFilter:

    def test_filter_by_section(self, store, sample_chunks):
        """Filtering by section should only return chunks from that section."""
        store.add_chunks(sample_chunks)
        results = store.search_with_filter("attention", section="Methods")
        for r in results:
            assert r["section"] == "Methods"

    def test_filter_by_has_math(self, store, sample_chunks):
        """Filtering by has_math=True should only return math chunks."""
        store.add_chunks(sample_chunks)
        results = store.search_with_filter("formula", has_math=True)
        for r in results:
            assert r["has_math"] is True

    def test_filter_with_no_matches_returns_empty(self, store, sample_chunks):
        """A filter that matches nothing should return []."""
        store.add_chunks(sample_chunks)
        # We have no chunks in "Conclusion" section
        results = store.search_with_filter("summary", section="Conclusion")
        assert results == []


class TestGetStats:

    def test_stats_on_empty_store(self, store):
        """Stats on an empty store should return zeros, not crash."""
        stats = store.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["sections"] == []

    def test_stats_counts_correctly(self, store, sample_chunks):
        """Stats should correctly count chunks, sections, math/code."""
        store.add_chunks(sample_chunks)
        stats = store.get_stats()
        assert stats["total_chunks"] == 5
        assert stats["math_chunks"] == 1
        assert stats["code_chunks"] == 1
        assert "Methods" in stats["sections"]
        assert "Abstract" in stats["sections"]

    def test_stats_sections_are_unique(self, store, sample_chunks):
        """Each section name should appear once in stats, even if multiple chunks share it."""
        store.add_chunks(sample_chunks)
        stats = store.get_stats()
        assert len(stats["sections"]) == len(set(stats["sections"]))


class TestClear:

    def test_clear_removes_all_chunks(self, store, sample_chunks):
        """After clear(), the store should be empty."""
        store.add_chunks(sample_chunks)
        assert store._collection.count() == 5
        store.clear()
        assert store._collection.count() == 0

    def test_can_add_after_clear(self, store, sample_chunks):
        """The store should be fully usable after a clear."""
        store.add_chunks(sample_chunks)
        store.clear()
        store.add_chunks(sample_chunks[:2])
        assert store._collection.count() == 2