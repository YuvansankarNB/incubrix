import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline.footage_manager import FootageManager

def main():
    fm = FootageManager()
    out = Path("scratch/test_pipeline_output.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    print("Rendering test procedural video with real character & internet background...")
    fm.generate_procedural_broll(
        query="hospital doctor consultation",
        duration=2.0,
        output_path=out,
        style="broadcast",
        character_type="human",
        scene_speaker="doctor",
        scene_text="We caught this early enough that targeted medication can fully reverse it.",
        visual_hint="hospital doctor consultation room"
    )
    print(f"Render complete! Output exists: {out.exists()} (size: {out.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
