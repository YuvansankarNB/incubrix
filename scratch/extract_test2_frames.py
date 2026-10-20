import cv2
from pathlib import Path

v_path = Path("data/storage/outputs/comp_test2_environmental_explainer_1792428573/video_16-9.mp4")
cap = cv2.VideoCapture(str(v_path))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out_dir = Path("scratch/verification_frames")
out_dir.mkdir(parents=True, exist_ok=True)

for p in [0.20, 0.50, 0.80]:
    idx = int(n_frames * p)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        out = out_dir / f"test2_environmental_frame_{idx}.jpg"
        cv2.imwrite(str(out), frame)
        print(f"Saved {out} (frame {idx}/{n_frames})")
cap.release()
