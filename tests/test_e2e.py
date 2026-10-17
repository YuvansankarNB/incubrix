import pytest
import time
from fastapi.testclient import TestClient
from src.main import app

def test_full_e2e_video_generation_and_clip_swap():
    with TestClient(app) as client:
        # 1. Submit job
        payload = {
            "script": "Welcome to IncuBrix. We create videos with artificial intelligence. Saving time is our core mission.",
            "aspect_ratios": ["16:9"],
            "style": "bright, modern, real-world footage",
            "avoid": ["people's faces"],
            "captions": {"burn_in": True, "srt": True}
        }
        
        headers = {"Idempotency-Key": f"e2e_test_{int(time.time())}"}
        response = client.post("/api/v1/broll/jobs", json=payload, headers=headers)
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        # 2. Poll until completion (up to 90s)
        max_wait = 90
        start = time.time()
        job_data = None

        while time.time() - start < max_wait:
            res = client.get(f"/api/v1/jobs/{job_id}")
            assert res.status_code == 200
            data = res.json()
            if data["status"] in ["succeeded", "failed"]:
                job_data = data
                break
            time.sleep(1)

        assert job_data is not None, "Job timed out"
        assert job_data["status"] == "succeeded", f"Job failed: {job_data.get('error')}"
        
        outputs = job_data["outputs"]
        assert "videos" in outputs
        assert "16:9" in outputs["videos"]
        assert "captions_srt" in outputs
        assert "timeline" in outputs
        assert "manifest" in outputs

        # 3. Test Clip Swapping endpoint
        swap_payload = {
            "scene_index": 0,
            "new_query": "futuristic data analytics screen",
            "preferred_source": "ai_gen"
        }
        swap_res = client.post(f"/api/v1/jobs/{job_id}/swap-clip", json=swap_payload)
        assert swap_res.status_code == 200
        swap_data = swap_res.json()
        assert swap_data["status"] == "succeeded"
        assert "videos" in swap_data["outputs"]
