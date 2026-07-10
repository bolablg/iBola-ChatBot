"""
Update the ChromaDB vector store with new, modified, or deleted documents.

Uses IntelligentChunker for section-based chunking with metadata enrichment.

Sync semantics (source-level upsert, not append-only):
  - State keys and chunk IDs use paths RELATIVE to the data directory, so the
    store survives checkout moves and different machines.
  - When a source file changes, its existing chunks are deleted before the new
    chunks are added (stable IDs: ``<relative_path>::<chunk_index>``).
  - When a source file disappears from the data directory, its chunks and
    state entry are removed.
  - ``--rebuild`` wipes the collection and state, then re-ingests everything.
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader

import config
from pipeline.chunker import IntelligentChunker
from utils.embedder import get_embeddings

VECTORSTORE_STATE_FILE = os.path.join(config.DB_PATH, ".vectorstore_state.json")


def get_vectorstore_state():
    """Load the last vectorstore state, normalizing legacy absolute paths.

    Older state files stored absolute paths from other checkouts, which made
    every file look new and duplicated chunks. Keys are normalized to paths
    relative to the data directory; unrecognizable keys are dropped.
    """
    if not os.path.exists(VECTORSTORE_STATE_FILE):
        return {}
    with open(VECTORSTORE_STATE_FILE, "r") as f:
        raw = json.load(f)

    state = {}
    for key, file_hash in raw.items():
        state[_normalize_state_key(key)] = file_hash
    return state


def _normalize_state_key(key):
    """Convert a legacy absolute path to a data-directory-relative path."""
    if not os.path.isabs(key):
        return key
    marker = os.sep + "data" + os.sep
    if marker in key:
        return key.split(marker, 1)[1]
    return os.path.basename(key)


def save_vectorstore_state(state):
    """Save the current vectorstore state to a local file."""
    with open(VECTORSTORE_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_file_hash(file_path):
    """Calculate the MD5 hash of a file."""
    hasher = hashlib.md5(usedforsecurity=False)
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def _load_documents(file_path):
    """Load a source file into raw LangChain documents, or None if unsupported."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        return None
    return loader.load()


def _delete_source_chunks(vectorstore, rel_path):
    """Delete all chunks belonging to a source file (by relative-path metadata)."""
    existing = vectorstore.get(where={"source": rel_path})
    ids = existing.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)
        print(f"  Deleted {len(ids)} stale chunks for {rel_path}")


def _scan_data_files():
    """Yield (relative_path, absolute_path) for every ingestible data file."""
    for root, _, files in os.walk(config.DATA_PATH):
        for file in files:
            if file.startswith(".") or file.endswith(".json"):
                continue
            abs_path = os.path.join(root, file)
            yield os.path.relpath(abs_path, config.DATA_PATH), abs_path


def update_vectorstore(rebuild=False):
    """Scan the data directory and upsert the vectorstore.

    Args:
        rebuild: wipe the collection and state, then re-ingest every file.
    """
    chunker = IntelligentChunker(min_words=50, max_words=800, overlap_words=100)
    vectorstore = Chroma(
        persist_directory=config.DB_PATH, embedding_function=get_embeddings()
    )

    if rebuild:
        print("Rebuilding vector store from scratch...")
        existing = vectorstore.get()
        if existing.get("ids"):
            vectorstore.delete(ids=existing["ids"])
            print(f"  Deleted all {len(existing['ids'])} existing chunks.")
        vectorstore_state = {}
    else:
        print("Scanning for new, modified, or deleted files...")
        vectorstore_state = get_vectorstore_state()

    updated = False
    seen = set()

    for rel_path, abs_path in _scan_data_files():
        seen.add(rel_path)
        file_hash = get_file_hash(abs_path)

        if vectorstore_state.get(rel_path) == file_hash:
            continue

        status = "New" if rel_path not in vectorstore_state else "Modified"
        print(f"{status} file detected: {rel_path}")

        try:
            raw_documents = _load_documents(abs_path)
            if raw_documents is None:
                print(f"Skipping unsupported file type: {rel_path}")
                continue

            all_chunks = []
            for doc in raw_documents:
                metadata = {**(doc.metadata or {}), "source": rel_path}
                all_chunks.extend(chunker.chunk_document(doc.page_content, metadata))

            if not all_chunks:
                print(f"  No chunks generated from {rel_path}")
                continue

            # Upsert: drop the previous chunks for this source, then add the
            # new ones under stable, path-relative IDs.
            _delete_source_chunks(vectorstore, rel_path)
            ids = [f"{rel_path}::{i}" for i in range(len(all_chunks))]
            vectorstore.add_documents(all_chunks, ids=ids)
            vectorstore_state[rel_path] = file_hash
            updated = True
            print(f"  Processed {len(all_chunks)} chunks from {rel_path}")

        except Exception as e:
            print(f"Error processing {rel_path}: {e}")

    # Remove chunks for source files that no longer exist on disk.
    for rel_path in list(vectorstore_state):
        if rel_path not in seen:
            print(f"Deleted file detected: {rel_path}")
            _delete_source_chunks(vectorstore, rel_path)
            del vectorstore_state[rel_path]
            updated = True

    if updated:
        save_vectorstore_state(vectorstore_state)
        print("Vector store updated.")
    else:
        print("No changes detected in vector store.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the ChromaDB vector store.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the collection and state, then re-ingest every data file.",
    )
    args = parser.parse_args()
    update_vectorstore(rebuild=args.rebuild)
