import sys
import asyncio
import cv2
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.job_queue import job_manager
from src.storage.s3_storage import storage

async def test_portrait_generation():
    job_id = f"comp_test_portrait_{int(time.time())}"
    print(f"Submitting 9:16 Portrait Test Job: {job_id}")
    
    payload = {
        "script": "Mobile first vertical video. Educational insights delivered with precision and clarity.",
        "character_type": "teacher",
        "aspect_ratios": ["9:16"],
        "research_mode": "script_only",
        "show_character_labels": False,
        "captions": {"burn_in": True, "srt": True}
    }

    job_state = job_manager.create_job(job_id, payload)
    
    # Wait for completion
    max_wait = 240
    t0 = time.time()
    while time.time() - t0 < max_wait:
        state = job_manager.get_job(job_id)
        if state:
            print(f"[Portrait] Progress: {state.progress}% | Step: {state.current_step} | Status: {state.status}")
            if state.status == "succeeded":
                break
            elif state.status == "failed":
                print(f"FAILED: {state.error}")
                return
        await asyncio.sleep(2)

    video_path = storage.get_job_dir(job_id) / "video_9-16.mp4"
    if not video_path.exists():
        print(f"Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Portrait video ready: {w}x{h}, {n_frames} frames")

    # Sample frame at 50%
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n_frames * 0.5))
    ret, frame = cap.read()
    if ret:
        out_frame = Path("scratch/verification_frames/test_portrait_frame.jpg")
        cv2.imwrite(str(out_frame), frame)
        print(f"Saved {out_frame}")
    cap.release()

if __name__ == "__main__":
    asyncio.run(test_portrait_generation())
