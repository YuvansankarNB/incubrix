import cv2
import json
from pathlib import Path

video_path = Path("data/storage/outputs/job_single_env_1792420674/video_16-9.mp4")
manifest_path = Path("data/storage/outputs/job_single_env_1792420674/manifest.json")
report_path = Path("data/storage/outputs/job_single_env_1792420674/research_report.md")

print(f"Video exists: {video_path.exists()} ({video_path.stat().st_size} bytes)")
print(f"Manifest exists: {manifest_path.exists()}")
print(f"Report exists: {report_path.exists()}")

if manifest_path.exists():
    mf = json.loads(manifest_path.read_text(encoding="utf-8"))
    print("\nManifest Research Provenance:")
    print(json.dumps(mf.get("research_provenance", {}), indent=2))

cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\nVideo Specs: {w}x{h} @ {fps} fps, {total_frames} frames ({total_frames/fps:.1f}s)")

# Extract 4 sample frames: 10%, 35%, 65%, 90%
for pct in [10, 35, 65, 90]:
    frame_idx = int(total_frames * (pct / 100.0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret and frame is not None:
        out_path = Path(f"scratch/env_test_frame_{pct}pct.png")
        cv2.imwrite(str(out_path), frame)
        print(f"Saved: {out_path} ({frame.shape})")

cap.release()
