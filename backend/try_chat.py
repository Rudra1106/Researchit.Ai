"""
try_chat.py — the full Phase 1 pipeline: PDF → chunks → store → chat.

This is your first working prototype. After this runs, you have a
system you can actually talk to about a research paper.

Usage: python backend/try_chat.py path/to/paper.pdf
"""

import sys
from pdf_processor   import process_pdf
from vector_store    import VectorStore
from llm_client      import chat, check_ollama_running

def main():
    if len(sys.argv) < 2:
        print("Usage: python try_chat.py <path_to_pdf>")
        sys.exit(1)

    # ── Startup check ──────────────────────────────────────────────────────────
    if not check_ollama_running():
        print("ERROR: Ollama is not running.")
        print("Start it with:  ollama serve")
        print("Then in another terminal: ollama pull llama3.2")
        sys.exit(1)
    print("Ollama is running.")

    # ── Load and process the PDF ───────────────────────────────────────────────
    pdf_path = sys.argv[1]
    print(f"\nProcessing PDF: {pdf_path}")
    chunks = process_pdf(pdf_path)
    print(f"Extracted {len(chunks)} chunks")

    # ── Store chunks in ChromaDB ───────────────────────────────────────────────
    print("\nStoring in vector store...")
    store = VectorStore()
    store.clear()
    store.add_chunks(chunks)
    stats = store.get_stats()
    print(f"Stored. Sections found: {stats['sections']}")

    # ── Chat loop ──────────────────────────────────────────────────────────────
    history = []
    print("\nReady! Ask questions about the paper. Type 'quit' to exit.\n")
    print("=" * 60)

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        # Retrieve relevant chunks
        chunks_for_question = store.search(question, n_results=3)

        if chunks_for_question:
            top = chunks_for_question[0]
            print(f"[Searching... found {len(chunks_for_question)} relevant sections, "
                  f"best match: '{top['section']}' p.{top['page']} "
                  f"score={top['score']:.2f}]")

        # Get the answer
        print("\nTutor: ", end="", flush=True)
        try:
            answer = chat(question, chunks_for_question, history)
            print(answer)

            # Save to history so next question has context
            history.append((question, answer))

            # Keep history to last 6 turns — beyond that the context window fills up
            if len(history) > 6:
                history.pop(0)

        except ConnectionError as e:
            print(f"\nERROR: {e}")
        except Exception as e:
            print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main()