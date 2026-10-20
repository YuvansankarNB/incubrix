import asyncio
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path("c:/Users/yuvan/Documents/Incubrix")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.job_queue import job_manager
from src.storage.s3_storage import storage

async def test_env():
    payload = {
        "script": "Urban tree canopies significantly lower city temperatures through natural shade. Planting native trees purifies urban air.",
        "character_type": "farmer",
        "research_mode": "script_plus_research",
        "is_voiceover": False,
        "aspect_ratios": ["16:9"],
        "style": "bright, modern, nature",
        "captions": {"burn_in": False, "srt": True}
    }
    job_id = f"job_single_env_{int(time.time())}"
    print(f"Creating job {job_id}...")
    t0 = time.time()
    state = job_manager.create_job(job_id, payload)
    
    last_step = ""
    while True:
        s = job_manager.get_job(job_id)
        if s.current_step != last_step:
            print(f"[{time.time()-t0:.1f}s] Step: {s.current_step} (Progress: {s.progress}%)")
            last_step = s.current_step
        if s.status in ["succeeded", "failed"]:
            print(f"[{time.time()-t0:.1f}s] Final status: {s.status}")
            if s.error:
                print(f"Error: {s.error}")
            else:
                print(f"Outputs: {s.outputs}")
            break
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(test_env())
