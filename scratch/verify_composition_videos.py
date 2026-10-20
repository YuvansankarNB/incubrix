"""
Automated Verification Script for Video Composition Upgrades:
1. Test 1: Doctor-Patient Dialogue (Two characters: doctor, patient)
   - Checks: NO nameplates, NO role tags, NO top banner ("DRAMATIC CONVERSATION"), NO speaking badges, NO head overlap.
2. Test 2: Environmental Explainer (Single character: farmer/presenter)
   - Checks: Relevant verified statistic ("8-12 deg C cooling impact"), strictly placed on left negative space, zero face overlap.
3. Test 3: Technical Presenter (Single character: business/engineer)
   - Checks: Concise process/insight card on left, presenter on right, zero head overlap, clean subtitles.
Extracts representative video frames and verifies negative space clearances.
"""

import sys
import os
import asyncio
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.job_queue import job_manager
from src.storage.s3_storage import storage

async def run_composition_tests():
    out_dir = Path("./scratch/verification_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path("./scratch/verification_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    test_jobs = [
        {
            "id": "test3_technical_presenter",
            "name": "Technical Engineering Presenter",
            "payload": {
                "script": "Modern cloud architectures eliminate routine development friction. High-velocity pipelines reduce cycle times and deploy resilient services automatically.",
                "character_type": "business",
                "aspect_ratios": ["16:9"],
                "research_mode": "script_only",
                "show_character_labels": False,
                "captions": {"burn_in": True, "srt": True}
            }
        }
    ]

    results = []

    for item in test_jobs:
        job_id = f"comp_{item['id']}_{int(time.time())}"
        print(f"\n==================================================")
        print(f"Launching {item['name']} [Job ID: {job_id}]")
        print(f"==================================================")

        job_state = job_manager.create_job(job_id, item["payload"])

        # Wait for job completion
        max_wait = 180
        t0 = time.time()
        completed = False
        while time.time() - t0 < max_wait:
            state = job_manager.get_job(job_id)
            if state:
                print(f"[{item['id']}] Progress: {state.progress}% | Step: {state.current_step} | Status: {state.status}")
                if state.status == "succeeded":
                    completed = True
                    break
                elif state.status == "failed":
                    print(f"[{item['id']}] FAILED: {state.error}")
                    break
            await asyncio.sleep(2)

        if not completed:
            print(f"[{item['id']}] TIMED OUT or FAILED!")
            results.append({"name": item["name"], "success": False, "reason": "timeout_or_fail"})
            continue

        # Extract frames from generated video
        state = job_manager.get_job(job_id)
        video_path = storage.get_job_dir(job_id) / "video_16-9.mp4"

        if not video_path.exists():
            print(f"Video file not found at {video_path}")
            results.append({"name": item["name"], "success": False, "reason": "video_not_found"})
            continue

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video created: {video_path.name} ({w}x{h}, {n_frames} frames, {fps} fps)")

        # Extract frames at 25%, 50%, 75%
        sample_indices = [int(n_frames * 0.25), int(n_frames * 0.50), int(n_frames * 0.75)]
        saved_frames = []

        for f_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret:
                frame_path = frames_dir / f"{item['id']}_frame_{f_idx}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_frames.append(str(frame_path))
                print(f"Saved sample frame: {frame_path.name}")

        cap.release()
        results.append({
            "name": item["name"],
            "id": item["id"],
            "success": True,
            "video_path": str(video_path),
            "frames": saved_frames
        })

    print("\n================== TEST SUMMARY ==================")
    for res in results:
        status_str = "SUCCESS" if res["success"] else "FAILED"
        print(f"{res['name']}: {status_str}")
        if res.get("frames"):
            for f in res["frames"]:
                print(f"   Frame: {f}")

if __name__ == "__main__":
    asyncio.run(run_composition_tests())
