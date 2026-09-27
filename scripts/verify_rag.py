import sys
import os
import asyncio

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.embeddings import embed

def test_embeddings():
    print("Testing embeddings with nomic-embed-text...")
    texts = ["Hello world", "RAG is awesome"]
    try:
        embeddings = asyncio.run(embed(texts))
        print(f"Successfully generated {len(embeddings)} embeddings.")
        print(f"First embedding size: {len(embeddings[0])}")
        return True
    except Exception as e:
        print(f"Embedding failed: {e}")
        return False

if __name__ == "__main__":
    if test_embeddings():
        print("RAG Verification PASSED")
    else:
        print("RAG Verification FAILED")
        sys.exit(1)
