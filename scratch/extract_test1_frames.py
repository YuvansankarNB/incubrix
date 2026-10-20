import cv2
from pathlib import Path

v_path = Path("data/storage/outputs/comp_test1_doctor_patient_dialogue_1792428190/video_16-9.mp4")
cap = cv2.VideoCapture(str(v_path))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out_dir = Path("scratch/verification_frames")
out_dir.mkdir(parents=True, exist_ok=True)

for p in [0.20, 0.45, 0.70]:
    idx = int(n_frames * p)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        out = out_dir / f"test1_dialogue_p{int(p*100)}.jpg"
        cv2.imwrite(str(out), frame)
        print(f"Saved {out} (frame {idx}/{n_frames})")
cap.release()
