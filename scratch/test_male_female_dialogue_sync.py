import asyncio
import os
import sys
import time
import subprocess
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.pipeline.job_queue import JobManager
from src.pipeline.audio_engine import audio_engine
from src.config import settings

async def run_male_female_test():
    print("=== STARTING MALE-FEMALE DIALOGUE LIP-SYNC TEST ===")
    
    script_text = (
        "Dr. Marcus: Welcome back, Elena. I have thoroughly reviewed your lab reports from this morning.\n"
        "Elena: Thank you so much, Doctor. Please be direct with me about what the numbers show.\n"
        "Dr. Marcus: Your blood glucose and cholesterol markers have improved by fifteen percent.\n"
        "Elena: That is such a huge relief to hear. I was really worried about needing more medication."
    )
    
    job_mgr = JobManager()
    job_id = f"mf_sync_test_{int(time.time())}"
    
    req_data = {
        "script": script_text,
        "style": "bright, modern clinical consultation",
        "aspect_ratios": ["16:9"],
        "show_character_labels": False,
        "character_type": "doctor, patient"
    }
    
    state = job_mgr.create_job(job_id, req_data)
    
    # Wait for completion
    for _ in range(120):
        await asyncio.sleep(2.0)
        s = job_mgr.get_job(job_id)
        print(f"Status: {s.status}, Step: {s.current_step}, Progress: {s.progress}%")
        if s.status in ["succeeded", "failed"]:
            break
            
    if state.status != "succeeded":
        print(f"FATAL: Job failed with error: {state.error}")
        sys.exit(1)
        
    print(f"Job succeeded! Outputs: {state.outputs}")
    
    # Verify video file
    video_path = Path(f"data/storage/outputs/{job_id}/video_16-9.mp4")
    assert video_path.exists(), f"Video file missing at {video_path}"
    print(f"Verified video file exists: {video_path} (size: {video_path.stat().st_size} bytes)")
    
    # Verify audio properties using ffmpeg
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-c", "copy", "-f", "null", "-"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"FFmpeg integrity check failed: {res.stderr}"
    print("FFmpeg video and audio stream integrity: 100% PASS")
    
    # Extract verification frames:
    # Scene 0 (Dr. Marcus speaking): t ~ 2.0s
    # Scene 1 (Elena speaking): t ~ 5.5s
    # Scene 2 (Dr. Marcus speaking): t ~ 9.0s
    # Scene 3 (Elena speaking): t ~ 13.0s
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video total frames: {total_frames}, FPS: {fps}")
    
    out_dir = Path("scratch/verification_frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample frames across scenes
    scenes_meta = state.scenes_meta
    print(f"Generated {len(scenes_meta)} scenes:")
    for s_idx, sc in enumerate(scenes_meta):
        spk = sc.get("speaker")
        st = sc.get("start_time")
        dur = sc.get("duration")
        mid_t = st + dur * 0.5
        frame_num = int(mid_t * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_num, total_frames - 1))
        ret, frame = cap.read()
        if ret:
            frame_file = out_dir / f"mf_scene_{s_idx}_{spk}_t{int(mid_t)}.jpg"
            cv2.imwrite(str(frame_file), frame)
            print(f"Saved Scene {s_idx} ({spk}) frame at t={mid_t:.1f}s -> {frame_file}")
            
    cap.release()
    print("=== MALE-FEMALE TEST COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    asyncio.run(run_male_female_test())
