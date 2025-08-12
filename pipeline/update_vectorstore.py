import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hashlib
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_chroma import Chroma
from utils.embedder import get_embeddings
import config

VECTORSTORE_STATE_FILE = os.path.join(config.DB_PATH, ".vectorstore_state.json")

def get_vectorstore_state():
    """Loads the last vectorstore state from a local file."""
    if os.path.exists(VECTORSTORE_STATE_FILE):
        with open(VECTORSTORE_STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_vectorstore_state(state):
    """Saves the current vectorstore state to a local file."""
    with open(VECTORSTORE_STATE_FILE, "w") as f:
        json.dump(state, f)

def get_file_hash(file_path):
    """Calculates the MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def update_vectorstore():
    """Scans the data directory and updates the vectorstore with new or modified files."""
    print("Scanning for new or modified files...")
    vectorstore_state = get_vectorstore_state()
    updated = False
    for root, _, files in os.walk(config.DATA_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            file_hash = get_file_hash(file_path)

            if file_path not in vectorstore_state or vectorstore_state[file_path] != file_hash:
                print(f"Processing {file_path}...")
                try:
                    if file.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                    elif file.endswith(".docx"):
                        loader = Docx2txtLoader(file_path)
                    elif file.endswith(".txt"):
                        loader = TextLoader(file_path)
                    else:
                        continue

                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    texts = text_splitter.split_documents(documents)

                    embeddings = get_embeddings()
                    Chroma.from_documents(texts, embeddings, persist_directory=config.DB_PATH)
                    vectorstore_state[file_path] = file_hash
                    updated = True
                    print(f"Successfully processed and updated vectorstore for {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    if updated:
        save_vectorstore_state(vectorstore_state)
        print("Vector store updated.")
    else:
        print("No changes detected in vector store.")

if __name__ == "__main__":
    update_vectorstore()