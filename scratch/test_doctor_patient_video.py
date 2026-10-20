import asyncio
import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline.footage_manager import footage_manager

async def main():
    audio_file = Path("sample_doctor_patient_dialogue.mp3")
    if not audio_file.exists():
        print("sample_doctor_patient_dialogue.mp3 not found!")
        return
        
    print("Generating dialogue clip with doctor & patient...")
    clip = await footage_manager.get_clip_for_scene(
        scene_index=0,
        search_query="doctor patient consultation hospital room",
        duration=4.0,
        style="bright, modern",
        audio_path=audio_file,
        start_time=0.0,
        scene_text="Doctor: What symptoms have you been experiencing recently?",
        character_type="doctor, patient",
        scene_speaker="Doctor",
        visual_hint="Doctor speaking to patient in clinic"
    )
    print(f"Generated clip successfully at: {clip.local_path}")
    
    # Extract frames to inspect
    cap = cv2.VideoCapture(str(clip.local_path))
    for frame_no in [10, 30, 60, 90]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret:
            out_img = f"scratch/dialogue_frame_{frame_no}.png"
            cv2.imwrite(out_img, frame)
            print(f"Saved {out_img}")
    cap.release()

if __name__ == "__main__":
    asyncio.run(main())
