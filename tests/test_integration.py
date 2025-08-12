import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

with patch("app.agent.get_agent", return_value=MagicMock()), \
     patch("app.agent.generate_response", return_value={"answer": "mock_answer", "actions": []}):
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


def test_chat_endpoint_invalid_user_input():
    response = client.post(
        "/chat",
        json={"user_input": "", "session_id": "test_session"}
    )
    assert response.status_code == 422


def test_chat_endpoint_invalid_session_id():
    response = client.post(
        "/chat",
        json={"user_input": "hello", "session_id": ""}
    )
    assert response.status_code == 422


def test_chat_endpoint_user_input_too_long():
    long_input = "a" * 501
    response = client.post(
        "/chat",
        json={"user_input": long_input, "session_id": "test_session"}
    )
    assert response.status_code == 422
