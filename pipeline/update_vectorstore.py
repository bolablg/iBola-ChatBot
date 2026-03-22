"""
Update the ChromaDB vector store with new or modified documents.

Uses IntelligentChunker for section-based chunking with metadata enrichment,
replacing the naive RecursiveCharacterTextSplitter.
"""

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
    """Load the last vectorstore state from a local file."""
    if os.path.exists(VECTORSTORE_STATE_FILE):
        with open(VECTORSTORE_STATE_FILE, "r") as f:
            return json.load(f)
    return {}


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


def update_vectorstore():
    """Scan the data directory and update the vectorstore with new or modified files."""
    print("Scanning for new or modified files...")
    vectorstore_state = get_vectorstore_state()
    chunker = IntelligentChunker(min_words=50, max_words=800, overlap_words=100)
    updated = False

    for root, _, files in os.walk(config.DATA_PATH):
        for file in files:
            file_path = os.path.join(root, file)

            # Skip hidden files and JSON files
            if file.startswith(".") or file.endswith(".json"):
                continue

            file_hash = get_file_hash(file_path)

            if (
                file_path in vectorstore_state
                and vectorstore_state[file_path] == file_hash
            ):
                continue

            if file_path not in vectorstore_state:
                print(f"New file detected: {file_path}")
            else:
                print(f"Modified file detected: {file_path}")

            print(f"Processing {file_path}...")
            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                elif file.endswith(".txt"):
                    loader = TextLoader(file_path)
                else:
                    print(f"Skipping unsupported file type: {file_path}")
                    continue

                raw_documents = loader.load()

                # Use intelligent chunker instead of naive splitter
                all_chunks = []
                for doc in raw_documents:
                    metadata = {
                        "source": file_path,
                        **(doc.metadata or {}),
                    }
                    chunks = chunker.chunk_document(doc.page_content, metadata)
                    all_chunks.extend(chunks)

                if all_chunks:
                    embeddings = get_embeddings()
                    Chroma.from_documents(
                        all_chunks, embeddings, persist_directory=config.DB_PATH
                    )
                    vectorstore_state[file_path] = file_hash
                    updated = True
                    print(f"  Processed {len(all_chunks)} chunks from {file_path}")
                else:
                    print(f"  No chunks generated from {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    if updated:
        save_vectorstore_state(vectorstore_state)
        print("Vector store updated.")
    else:
        print("No changes detected in vector store.")


if __name__ == "__main__":
    update_vectorstore()
