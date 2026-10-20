import asyncio
import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline.footage_manager import footage_manager

async def main():
    audio_file = Path("sample_routine_voiceover.mp3")
    print("Generating single character clip with teacher...")
    clip = await footage_manager.get_clip_for_scene(
        scene_index=1,
        search_query="classroom educational lesson teacher whiteboard",
        duration=3.0,
        style="bright, modern",
        audio_path=audio_file,
        start_time=0.0,
        scene_text="In today's lesson, we explore how sustainable systems transform modern communities.",
        character_type="teacher",
        scene_speaker="Teacher",
        visual_hint="Teacher explaining on whiteboard"
    )
    print(f"Generated clip successfully at: {clip.local_path}")
    
    cap = cv2.VideoCapture(str(clip.local_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("scratch/teacher_frame_25.png", frame)
        print("Saved scratch/teacher_frame_25.png")
    cap.release()

if __name__ == "__main__":
    asyncio.run(main())
