import io
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

fake_uv = ModuleType("pipeline.update_vectorstore")
fake_uv.update_vectorstore = lambda: None
sys.modules["pipeline.update_vectorstore"] = fake_uv

from pipeline import sync


class DummyDownload:
    """Minimal stand-in for MediaIoBaseDownload."""

    def __init__(self, fh, request):
        self.fh = fh
        self.request = request

    def next_chunk(self):
        self.fh.write(self.request.read())
        status = SimpleNamespace(progress=lambda: 1.0)
        return status, True


class MockRequest(io.BytesIO):
    """Simple request object that returns predefined bytes."""

    def read(self, *args, **kwargs):  # pylint: disable=unused-argument
        return super().getvalue()


def test_sync_folder_writes_files(tmp_path, monkeypatch):
    """sync_folder_recursive downloads files to the local path."""

    item = {
        "id": "1",
        "name": "example.txt",
        "mimeType": "text/plain",
        "modifiedTime": "2024-01-01T00:00:00Z",
    }

    service = MagicMock()
    files_resource = MagicMock()
    service.files.return_value = files_resource
    files_resource.list.return_value.execute.return_value = {"files": [item]}
    files_resource.get_media.return_value = MockRequest(b"content")

    monkeypatch.setattr(sync, "MediaIoBaseDownload", DummyDownload)

    sync_state = {}
    updated = sync.sync_folder_recursive(service, "folder", tmp_path, sync_state)

    downloaded = (tmp_path / "example.txt").read_bytes()
    assert downloaded == b"content"
    assert sync_state[item["id"]] == item["modifiedTime"]
    assert updated == [f"New: {item['name']} (Modified: {item['modifiedTime']})"]
