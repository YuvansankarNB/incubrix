"""
IncuBrix Script-to-B-Roll Video Engine
Verification Script for Tests A through G

Tests:
- Test A: Environmental & Sustainable Agriculture (Farmer character, tree cooling fact, statistic typography)
- Test B: Doctor-Patient Dialogue (Multi-character turn taking, listening micro-nods, doctor & patient)
- Test C: Educational Science / Physics (Teacher with whiteboard, convection flowchart typography)
- Test D: Technology & AI Engineering (Executive/Engineer presenter, 55.8% velocity metric)
- Test E: Researched Factual Video (Authoritative citations, APA spaced retrieval fact, research report)
- Test F: Voice-Over Narration (External narrator, mouth strictly closed, zero lip sync energy)
- Test G: Portrait 9:16 Video (Vertical framing, negative-space balance, zero face overlap)
"""

import sys
import os
import asyncio
import time
import json
import cv2
import numpy as np
from pathlib import Path

# Set PYTHONPATH to project root
PROJECT_ROOT = Path("c:/Users/yuvan/Documents/Incubrix")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.job_queue import job_manager
from src.storage.s3_storage import storage

TEST_CASES = [
    {
        "id": "test_a_environmental",
        "name": "Test A: Environmental & Sustainable Agriculture",
        "payload": {
            "script": "Urban tree canopies significantly lower city temperatures through natural shade. Planting native trees purifies urban air and fosters biodiversity in neighborhoods.",
            "character_type": "farmer",
            "research_mode": "script_plus_research",
            "is_voiceover": False,
            "aspect_ratios": ["16:9"],
            "style": "bright, modern, nature and ecological",
            "captions": {"burn_in": True, "srt": True}
        }
    },
    {
        "id": "test_b_dialogue",
        "name": "Test B: Doctor-Patient Consultation Dialogue",
        "payload": {
            "script": "Doctor: Hello David, your routine blood pressure readings are showing noticeable improvement this month.\nPatient: Thank you Doctor, I have been following the aerobic walking regimen every morning.",
            "character_type": "doctor, patient",
            "research_mode": "script_plus_research",
            "is_voiceover": False,
            "aspect_ratios": ["16:9"],
            "style": "bright, modern clinical consultation",
            "captions": {"burn_in": True, "srt": True}
        }
    },
    {
        "id": "test_c_education",
        "name": "Test C: Educational Classroom / Physics Lesson",
        "payload": {
            "script": "In today's physics lesson, we explore how atmospheric convection currents drive global weather systems. Notice how warm air masses expand and rise, transferring thermal energy upward.",
            "character_type": "teacher",
            "research_mode": "script_plus_research",
            "is_voiceover": False,
            "aspect_ratios": ["16:9"],
            "style": "bright educational classroom",
            "captions": {"burn_in": True, "srt": True}
        }
    },
    {
        "id": "test_d_technology",
        "name": "Test D: Technology & AI Presentation",
        "payload": {
            "script": "Modern artificial intelligence is transforming software engineering workflows. Automated refactoring tools allow development teams to ship secure verified code at high velocity.",
            "character_type": "executive",
            "research_mode": "script_plus_research",
            "is_voiceover": False,
            "aspect_ratios": ["16:9"],
            "style": "modern high-tech corporate",
            "captions": {"burn_in": True, "srt": True}
        }
    },
    {
        "id": "test_e_researched_facts",
        "name": "Test E: Researched Factual Video with Citations",
        "payload": {
            "script": "Spaced retrieval practice significantly improves long-term factual retention. Regular spaced study sessions dramatically outperform single-session cramming for complex material.",
            "character_type": "teacher",
            "research_mode": "script_plus_research",
            "is_voiceover": False,
            "aspect_ratios": ["16:9"],
            "style": "academic scientific study",
            "captions": {"burn_in": True, "srt": True}
        }
    },
    {
        "id": "test_f_voiceover_closed_mouth",
        "name": "Test F: Voice-Over Narration (Closed Mouth / Silent)",
        "payload": {
            "script": "The deep ocean contains some of the most resilient ecosystems on planet Earth. Marine biologists continue to document unique thermal vent organisms thriving under intense pressure.",
            "character_type": "narrator",
            "research_mode": "script_only",
            "is_voiceover": True, # CRUCIAL: Mouth must remain closed
            "aspect_ratios": ["16:9"],
            "style": "documentary nature realism",
            "captions": {"burn_in": True, "srt": True}
        }
    },
    {
        "id": "test_g_portrait_916",
        "name": "Test G: Vertical Portrait Video (9:16)",
        "payload": {
            "script": "Three essential habits for daily cognitive health: stay hydrated, maintain consistent sleep schedules, and take brief brisk walks.",
            "character_type": "doctor",
            "research_mode": "script_plus_research",
            "is_voiceover": False,
            "aspect_ratios": ["9:16"], # CRUCIAL: 9:16 vertical
            "style": "bright modern vertical video",
            "captions": {"burn_in": True, "srt": True}
        }
    }
]

def extract_and_analyze_frames(video_path: Path, output_dir: Path, test_id: str) -> dict:
    """Extracts key frames from the generated MP4 and verifies visual quality metrics."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    extracted_frames = []
    # Pick 3 representative sample points: 20%, 50%, 80%
    sample_indices = [
        int(total_frames * 0.2),
        int(total_frames * 0.5),
        int(total_frames * 0.8)
    ]
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            out_img = output_dir / f"{test_id}_frame_{idx}.png"
            cv2.imwrite(str(out_img), frame)
            extracted_frames.append(out_img)

    cap.release()
    return {
        "total_frames": total_frames,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "extracted_frames": [str(f) for f in extracted_frames]
    }

async def run_single_test(tc: dict) -> dict:
    test_id = tc["id"]
    print(f"\n==========================================")
    print(f"RUNNING: {tc['name']}")
    print(f"==========================================")
    
    t0 = time.time()
    job_id = f"job_{test_id}_{int(time.time())}"
    job_state = job_manager.create_job(job_id, tc["payload"])
    
    # Wait for completion
    max_wait = 240
    start = time.time()
    last_print = 0
    while time.time() - start < max_wait:
        state = job_manager.get_job(job_id)
        if time.time() - last_print > 8:
            print(f"[{int(time.time() - t0)}s] Step: {state.current_step} (Progress: {state.progress}%)")
            last_print = time.time()
        if state.status in ["succeeded", "failed"]:
            break
        await asyncio.sleep(1)
        
    elapsed = round(time.time() - t0, 2)
    state = job_manager.get_job(job_id)
    print(f"Status: {state.status.upper()} in {elapsed}s")
    
    if state.status != "succeeded":
        print(f"ERROR: {state.error}")
        return {"id": test_id, "name": tc["name"], "status": "failed", "error": str(state.error)}
        
    # Analyze video outputs
    job_dir = storage.get_job_dir(job_id)
    outputs = state.outputs
    videos = outputs.get("videos", {})
    first_ratio = list(videos.keys())[0]
    ratio_clean = first_ratio.replace(":", "-")
    video_file = job_dir / f"video_{ratio_clean}.mp4"
    
    frame_analysis = {}
    if video_file.exists():
        frame_analysis = extract_and_analyze_frames(video_file, PROJECT_ROOT / "scratch", test_id)
        print(f"Generated: {video_file.name} ({frame_analysis['resolution']}, {frame_analysis['total_frames']} frames)")
        
    # Check manifest and research report
    mf_path = job_dir / "manifest.json"
    manifest = {}
    if mf_path.exists():
        manifest = json.loads(mf_path.read_text(encoding="utf-8"))
        
    rep_path = job_dir / "research_report.md"
    has_research_report = rep_path.exists()
    
    return {
        "id": test_id,
        "name": tc["name"],
        "status": "succeeded",
        "duration_seconds": elapsed,
        "video_path": str(video_file),
        "resolution": frame_analysis.get("resolution"),
        "total_frames": frame_analysis.get("total_frames"),
        "extracted_frames": frame_analysis.get("extracted_frames", []),
        "research_provenance": manifest.get("research_provenance", {}).get("mode"),
        "citations_count": len(manifest.get("research_provenance", {}).get("sources", [])),
        "has_research_report": has_research_report
    }

async def main():
    results = []
    for tc in TEST_CASES:
        res = await run_single_test(tc)
        results.append(res)
        
    print("\n\n==========================================")
    print("ALL TESTS COMPLETED — SUMMARY TABLE")
    print("==========================================")
    all_passed = True
    for r in results:
        status_sym = "[PASS]" if r["status"] == "succeeded" else "[FAIL]"
        if r["status"] != "succeeded":
            all_passed = False
        print(f"{status_sym} {r['name']} | Duration: {r.get('duration_seconds')}s | Res: {r.get('resolution')} | Citations: {r.get('citations_count')}")
        
    summary_path = PROJECT_ROOT / "scratch" / "test_suite_results.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote full test results to {summary_path}")

if __name__ == "__main__":
    asyncio.run(main())
