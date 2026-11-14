import io
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from behavior_detection.config import load_config
from behavior_detection.web import create_app


class FakeDetector:
    def predict(self, frame, iou=0.45):
        height, width = frame.shape[:2]
        return {"width": width, "height": height, "inference_ms": 1,
                "detections": [{"id": 0, "class_id": 2, "class_name": "hand_raised",
                                "confidence": 0.85, "box": [0.1, 0.2, 0.4, 0.7]}]}


@pytest.fixture
def client():
    with TestClient(create_app(load_config(), detector=FakeDetector())) as session:
        yield session


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (80, 40), "red").save(buffer, "PNG")
    return buffer.getvalue()


def video_bytes(tmp_path, frames=12):
    path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12, (80, 48))
    assert writer.isOpened()
    for i in range(frames):
        writer.write(np.full((48, 80, 3), i % 255, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def wait_for_job(client, key):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = client.get(f"/api/videos/{key}")
        assert response.status_code == 200
        status = response.json()
        if status["state"] in {"complete", "cancelled", "error"}:
            return status
        time.sleep(0.05)
    pytest.fail("Video job did not finish")


def test_image_returns_normalized_boxes_and_rejects_invalid_inputs(client, monkeypatch):
    response = client.post("/api/image", files={"file": ("test.png", image_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["width"] == 80
    assert response.json()["detections"][0]["box"] == [0.1, 0.2, 0.4, 0.7]
    assert client.post("/api/image", files={"file": ("bad.png", b"not an image")}).status_code == 422
    assert client.post("/api/image?iou=3", files={"file": ("test.png", image_bytes())}).status_code == 422
    monkeypatch.setattr("behavior_detection.web.IMAGE_LIMIT", 8)
    assert client.post("/api/image", files={"file": ("test.png", image_bytes())}).status_code == 413


def test_metadata_uses_config_order_and_unknown_routes_are_404(client):
    assert [c["name"] for c in client.get("/api/meta").json()["classes"]] == load_config()["classes"]
    assert client.get("/api/videos/not-a-job/results").status_code == 404
    assert client.get("/api/missing").status_code == 404


def test_video_is_sampled_encoded_and_seekable(client, tmp_path):
    response = client.post("/api/videos?sample_fps=4", files={"file": ("input.avi", video_bytes(tmp_path))})
    assert response.status_code == 202
    key = response.json()["id"]
    status = wait_for_job(client, key)
    assert status["state"] == "complete", status
    result = client.get(f"/api/videos/{key}/results").json()
    assert len(result["frames"]) == 4
    assert [f["time"] for f in result["frames"]] == [0, 0.25, 0.5, 0.75]
    assert result["duration"] == 1
    assert result["audio"] is False
    preview = client.get(f"/api/videos/{key}/preview.mp4", headers={"Range": "bytes=0-99"})
    assert preview.status_code == 206
    assert len(preview.content) == 100
    assert client.get(f"/api/videos/{key}/annotated.mp4").status_code == 200
    assert not list((client.app.state.temp / key).glob("source*"))


def test_bad_video_fails_without_leaving_partial_exports(client):
    response = client.post("/api/videos", files={"file": ("bad.mp4", b"invalid")})
    key = response.json()["id"]
    assert wait_for_job(client, key)["state"] == "error"
    assert client.get(f"/api/videos/{key}/results").status_code == 409
    assert client.post("/api/videos", files={"file": ("bad.exe", b"invalid")}).status_code == 415


def test_video_can_be_cancelled_and_temporary_files_are_removed(tmp_path):
    class SlowDetector(FakeDetector):
        def predict(self, frame, iou=0.45):
            time.sleep(0.15)
            return super().predict(frame, iou)
    with TestClient(create_app(load_config(), detector=SlowDetector())) as client:
        temp = client.app.state.temp
        response = client.post("/api/videos", files={"file": ("input.avi", video_bytes(tmp_path, 120))})
        key = response.json()["id"]
        assert client.delete(f"/api/videos/{key}").status_code == 200
        assert wait_for_job(client, key)["state"] == "cancelled"
    assert not temp.exists()
