import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200

def test_chat_endpoint_in_scope():
    response = client.post(
        "/chat",
        json={"user_input": "what is your experience?", "session_id": "test_session"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "actions" in data

def test_chat_endpoint_out_of_scope():
    response = client.post(
        "/chat",
        json={"user_input": "what is the weather like?", "session_id": "test_session"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "actions" in data
    assert data["answer"] == "I am trained to answer questions about Bolaji's professional background. Please ask a relevant question or contact Bolaji directly."
