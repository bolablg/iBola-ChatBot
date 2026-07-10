"""
PART 7: feedback score schema, PII masking, and trace-id hygiene.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.services.tracing import hash_visitor, mask_pii


class TestPIIMasking:
    def test_email_phone_ip_url_masked(self):
        text = (
            "Reach me at bolaji@example.com or +229 0196 653 056, "
            "server 192.168.1.42, docs at https://secret.internal/x"
        )
        out = mask_pii(text)
        assert "bolaji@example.com" not in out
        assert "192.168.1.42" not in out
        assert "https://secret.internal/x" not in out
        assert "[email]" in out and "[ip]" in out and "[url]" in out

    def test_clean_text_unchanged(self):
        text = "Bolaji was Data Director at Gozem through July 2026."
        assert mask_pii(text) == text

    def test_dates_and_metrics_not_masked(self):
        # The phone regex must not eat dates or KB numbers (Codex)
        for keep in ("2026-07-10", "42.57%", "650+ people", "0 to 14+", "88% match"):
            assert keep.split()[0] in mask_pii(f"value {keep} here")

    def test_masks_nested_payload(self):
        payload = {"rewritten_query": "email me at x@y.com", "n": 5}
        masked = mask_pii(payload)
        assert "x@y.com" not in masked["rewritten_query"]
        assert masked["n"] == 5

    def test_hash_visitor_is_anonymous_and_stable(self):
        a = hash_visitor("1.2.3.4", "Mozilla/5.0")
        b = hash_visitor("1.2.3.4", "Mozilla/5.0")
        assert a == b and a.startswith("anon-")
        assert "1.2.3.4" not in a
        assert hash_visitor(None, None) is None


class TestFeedbackEndpoint:
    @pytest.fixture
    def client(self):
        with patch("app.routes.feedback.get_tracer") as mock_tracer:
            tracer = Mock()
            mock_tracer.return_value = tracer
            from app.main import app

            yield TestClient(app), tracer

    def test_thumbs_down_records_boolean_trace_score(self, client):
        c, tracer = client
        r = c.post(
            "/feedback",
            json={
                "score_name": "user-thumbs",
                "value": 0,
                "session_id": "s1",
                "trace_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
            },
        )
        assert r.status_code == 200
        tracer.score_trace.assert_called_once()
        kwargs = tracer.score_trace.call_args.kwargs
        assert kwargs["name"] == "user-thumbs"
        assert kwargs["data_type"] == "BOOLEAN"
        assert kwargs["trace_id"] == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"

    def test_reason_is_categorical(self, client):
        c, tracer = client
        r = c.post(
            "/feedback",
            json={
                "score_name": "user-thumbs-reason",
                "value": "too-vague",
                "session_id": "s1",
                "trace_id": "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f",
            },
        )
        assert r.status_code == 200
        assert tracer.score_trace.call_args.kwargs["data_type"] == "CATEGORICAL"

    def test_csat_is_session_level(self, client):
        c, tracer = client
        r = c.post(
            "/feedback",
            json={"score_name": "session-csat", "value": "3", "session_id": "s1"},
        )
        assert r.status_code == 200
        tracer.score_session.assert_called_once()
        tracer.score_trace.assert_not_called()

    def test_unknown_score_name_rejected(self, client):
        c, _ = client
        r = c.post(
            "/feedback",
            json={"score_name": "made-up", "value": 1, "session_id": "s1"},
        )
        assert r.status_code == 400

    def test_unknown_reason_rejected(self, client):
        c, _ = client
        r = c.post(
            "/feedback",
            json={
                "score_name": "user-thumbs-reason",
                "value": "nonsense",
                "session_id": "s1",
                "trace_id": "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f",
            },
        )
        assert r.status_code == 400

    def test_trace_score_requires_trace_id(self, client):
        c, _ = client
        r = c.post(
            "/feedback",
            json={"score_name": "user-thumbs", "value": 1, "session_id": "s1"},
        )
        assert r.status_code == 400

    def test_boolean_rejects_non_binary(self, client):
        c, _ = client
        r = c.post(
            "/feedback",
            json={
                "score_name": "user-thumbs",
                "value": 5,
                "session_id": "s1",
                "trace_id": "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f",
            },
        )
        assert r.status_code == 400
