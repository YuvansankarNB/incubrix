#!/usr/bin/env python3
"""
IncuBrix Batch Runner CLI
Integration Standard #10: Runs a folder or file of test inputs against the API and writes all results to one folder.
"""

import os
import sys
import time
import json
import argparse
import urllib.request
import urllib.parse
from pathlib import Path

def submit_job(api_base: str, payload: dict) -> str:
    url = f"{api_base}/api/v1/broll/jobs"
    req_body = json.dumps(payload).encode('utf-8')
    idempotency_key = f"batch_{int(time.time())}_{abs(hash(req_body)) % 100000}"
    
    req = urllib.request.Request(
        url,
        data=req_body,
        headers={
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key
        }
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        return data["job_id"]

def poll_job(api_base: str, job_id: str, timeout: int = 180) -> dict:
    url = f"{api_base}/api/v1/jobs/{job_id}"
    start = time.time()
    while time.time() - start < timeout:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            status = data.get("status")
            if status == "succeeded":
                return data
            elif status == "failed":
                raise RuntimeError(f"Job failed: {data.get('error')}")
        time.sleep(1.5)
    raise TimeoutError(f"Job {job_id} timed out after {timeout} seconds")

def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))

def run_batch(input_path: Path, output_dir: Path, api_base: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify test inputs
    inputs = []
    if input_path.is_file():
        inputs.append(input_path)
    elif input_path.is_dir():
        inputs.extend(sorted(input_path.glob("*.json")))
        inputs.extend(sorted(input_path.glob("*.txt")))

    if not inputs:
        print(f"[!] No test files found in {input_path}")
        return

    print(f"[*] Starting batch runner on {len(inputs)} inputs against {api_base}")
    summary = []

    for idx, f in enumerate(inputs):
        test_name = f.stem
        target_subfolder = output_dir / test_name
        target_subfolder.mkdir(parents=True, exist_ok=True)
        print(f"\n[{idx+1}/{len(inputs)}] Processing: {test_name}")

        try:
            # Parse payload
            if f.suffix == ".json":
                with open(f, "r", encoding="utf-8") as jf:
                    payload = json.load(jf)
            else:
                with open(f, "r", encoding="utf-8") as tf:
                    payload = {"script": tf.read().strip(), "aspect_ratios": ["16:9", "9:16"]}

            # Submit and poll
            t0 = time.time()
            job_id = submit_job(api_base, payload)
            print(f"    -> Job submitted: {job_id}")
            result = poll_job(api_base, job_id)
            elapsed = round(time.time() - t0, 2)
            print(f"    -> Job succeeded in {elapsed}s")

            # Save deliverables to output directory
            outputs = result.get("outputs", {})
            for ratio, vid_url in outputs.get("videos", {}).items():
                dest_vid = target_subfolder / f"video_{ratio.replace(':', '-')}.mp4"
                download_file(vid_url, dest_vid)

            if outputs.get("captions_srt"):
                download_file(outputs["captions_srt"], target_subfolder / "captions.srt")
            if outputs.get("timeline"):
                download_file(outputs["timeline"], target_subfolder / "timeline.json")
            if outputs.get("manifest"):
                download_file(outputs["manifest"], target_subfolder / "manifest.json")

            summary.append({
                "test": test_name,
                "job_id": job_id,
                "status": "PASS",
                "duration_seconds": elapsed,
                "saved_to": str(target_subfolder)
            })

        except Exception as e:
            print(f"    [X] Error: {str(e)}")
            summary.append({
                "test": test_name,
                "status": "FAIL",
                "error": str(e)
            })

    # Write batch summary
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    print(f"\n[✓] Batch runner finished. Summary written to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IncuBrix CLI Batch Test Runner")
    parser.add_argument("--input", "-i", type=str, default="./test_pack/inputs", help="Input file or folder")
    parser.add_argument("--output", "-o", type=str, default="./test_pack/outputs", help="Output directory")
    parser.add_argument("--api", "-a", type=str, default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    run_batch(Path(args.input), Path(args.output), args.api)
