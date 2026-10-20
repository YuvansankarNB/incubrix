import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline.footage_manager import FootageManager
import cv2

def main():
    fm = FootageManager()
    out = Path("scratch/test_drama_output.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    print("Rendering two-speaker drama dialogue...")
    fm.generate_procedural_broll(
        query="hospital ward doctor patient",
        duration=2.0,
        output_path=out,
        style="broadcast",
        character_type="human",
        scene_speaker="doctor",
        scene_text="Doctor: The latest lab reports just came in.\nPatient: Does this mean I need immediate surgery?",
        visual_hint="hospital examination room"
    )
    print("Drama render complete!")
    cap = cv2.VideoCapture(str(out))
    cap.set(cv2.CAP_PROP_POS_MSEC, 800)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("scratch/test_drama_frame.png", frame)
        print("Saved scratch/test_drama_frame.png")
    cap.release()

if __name__ == "__main__":
    main()
