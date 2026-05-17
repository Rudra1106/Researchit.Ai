"""
vector_store.py

Responsibility: store chunk embeddings and retrieve relevant chunks by query.

This module knows nothing about PDFs or agents. It only answers:
  "Store this text." and "What stored texts are most similar to this query?"

We use:
  - sentence-transformers  →  converts text to vectors (runs locally)
  - ChromaDB               →  stores vectors and does similarity search
"""

import chromadb
from sentence_transformers import SentenceTransformer


# ── Constants ──────────────────────────────────────────────────────────────────

# This model is small (80MB), fast on CPU, and good enough for retrieval.
# It produces 384-dimensional vectors.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ChromaDB will store its files in this folder inside your project.
# You can delete this folder to reset everything.
CHROMA_DB_PATH = "./chroma_db"

# Name of the collection inside ChromaDB.
# Think of it like a table name in a regular database.
COLLECTION_NAME = "paper_chunks"

# How many chunks to return by default when searching.
DEFAULT_N_RESULTS = 5


# ── VectorStore class ──────────────────────────────────────────────────────────

class VectorStore:
    """
    Wraps ChromaDB + a sentence-transformer embedding model.

    Typical usage:
        store = VectorStore()
        store.add_chunks(chunks)          # from pdf_processor
        results = store.search("what is softmax?")
        for r in results:
            print(r["text"], r["section"])
    """

    def __init__(self, db_path=CHROMA_DB_PATH, collection_name=COLLECTION_NAME):
        # Load the embedding model once. This takes ~2 seconds on first run
        # (downloads the model), then is instant because it's cached locally.
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # PersistentClient saves data to disk so it survives between runs.
        # Use EphemeralClient() instead if you want in-memory only (good for tests).
        self._client = chromadb.PersistentClient(path=db_path)

        # get_or_create: if the collection already exists, open it; otherwise make it.
        # We set the distance metric to cosine similarity — standard for text.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Vector store ready. Chunks stored: {self._collection.count()}")

    # ── Write operations ───────────────────────────────────────────────────────

    def add_chunks(self, chunks):
        """
        Embed and store a list of chunks from pdf_processor.

        ChromaDB's add() takes parallel lists:
          ids        = ["0", "1", "2", ...]
          embeddings = [[0.23, ...], [0.11, ...], ...]
          documents  = ["chunk text...", ...]
          metadatas  = [{"section": "...", "page": 1, ...}, ...]

        We batch everything and send it in one call — much faster than
        adding chunks one by one.
        """
        if not chunks:
            print("No chunks to add.")
            return

        print(f"Embedding {len(chunks)} chunks...")

        # Extract just the text for batch embedding.
        # embed() processes all texts in one go — efficient.
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self._model.encode(texts, show_progress_bar=True)
        # embeddings is now a numpy array of shape (len(chunks), 384)

        # ChromaDB wants plain Python lists, not numpy arrays.
        embeddings_list = embeddings.tolist()

        ids = [str(chunk["chunk_id"]) for chunk in chunks]

        # Metadata: only store what we'll actually filter or display.
        # ChromaDB metadata values must be str, int, float, or bool.
        metadatas = [
            {
                "section":  chunk["section"],
                "page":     chunk["page"],
                "has_math": chunk["has_math"],
                "has_code": chunk["has_code"],
                "word_count": chunk["word_count"],
            }
            for chunk in chunks
        ]

        # upsert = insert if new, update if id already exists.
        # This is safer than add() which throws an error on duplicate ids.
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"Stored {len(chunks)} chunks. Total in store: {self._collection.count()}")

    def clear(self):
        """
        Remove all chunks from the store.
        Call this before loading a new paper so old chunks don't pollute results.
        """
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("Vector store cleared.")

    # ── Read operations ────────────────────────────────────────────────────────

    def search(self, query, n_results=DEFAULT_N_RESULTS, where=None):
        """
        Find the most relevant chunks for a query string.

        Args:
            query:     natural language question or keyword string
            n_results: how many chunks to return
            where:     optional ChromaDB metadata filter, e.g.
                       {"has_math": True} to only return math chunks
                       {"section": "Methods"} to only search one section

        Returns:
            list of dicts, each with: text, section, page, has_math,
            has_code, word_count, score, chunk_id
            Sorted by relevance (most relevant first).
        """
        if self._collection.count() == 0:
            print("Warning: vector store is empty. Did you call add_chunks()?")
            return []

        # Embed the query with the same model used for chunks.
        # IMPORTANT: always use the same model for queries and documents.
        query_embedding = self._model.encode(query).tolist()

        # Build kwargs — only add 'where' if it was provided
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        # ChromaDB returns nested lists (one per query, we only sent one query).
        # Unpack the first (and only) query's results.
        results = []
        for i in range(len(raw["ids"][0])):
            results.append({
                "chunk_id": raw["ids"][0][i],
                "text":     raw["documents"][0][i],
                "section":  raw["metadatas"][0][i]["section"],
                "page":     raw["metadatas"][0][i]["page"],
                "has_math": raw["metadatas"][0][i]["has_math"],
                "has_code": raw["metadatas"][0][i]["has_code"],
                "score":    1 - raw["distances"][0][i],
                # score = 1 - cosine_distance. Range: 0 (unrelated) to 1 (identical).
            })

        return results

    def search_with_filter(self, query, section=None, has_math=None, n_results=DEFAULT_N_RESULTS):
        """
        Convenience wrapper around search() for common filter combinations.

        Example:
            # Only search in the Methods section for math-heavy chunks
            results = store.search_with_filter(
                "gradient computation",
                section="Methods",
                has_math=True
            )
        """
        where = {}
        if section is not None:
            where["section"] = section
        if has_math is not None:
            where["has_math"] = has_math

        return self.search(query, n_results=n_results, where=where if where else None)

    def get_stats(self):
        """
        Return basic stats about what's stored.
        Useful for debugging — call this after add_chunks() to verify.
        """
        count = self._collection.count()
        if count == 0:
            return {"total_chunks": 0, "sections": []}

        # Fetch all metadata to compute stats
        all_data = self._collection.get(include=["metadatas"])
        sections = list({m["section"] for m in all_data["metadatas"]})
        math_chunks = sum(1 for m in all_data["metadatas"] if m["has_math"])
        code_chunks = sum(1 for m in all_data["metadatas"] if m["has_code"])

        return {
            "total_chunks": count,
            "sections": sorted(sections),
            "math_chunks": math_chunks,
            "code_chunks": code_chunks,
        }