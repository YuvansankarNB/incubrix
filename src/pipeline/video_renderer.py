import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import imageio_ffmpeg
from src.config import settings
from src.logging_config import logger

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

ASPECT_CONFIGS = {
    "16:9": {"width": 1920, "height": 1080, "name": "16-9"},
    "9:16": {"width": 1080, "height": 1920, "name": "9-16"},
    "1:1":  {"width": 1080, "height": 1080, "name": "1-1"}
}

def format_srt_time(seconds: float) -> str:
    millis = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"

class VideoRenderer:
    def generate_srt(self, scenes: List[Dict[str, Any]], output_path: Path):
        """Generates standard .srt caption file from scenes."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for i, s in enumerate(scenes):
            start_str = format_srt_time(s["start_time"])
            end_str = format_srt_time(s["end_time"])
            text = s["narration"].strip()
            lines.append(f"{i+1}\n{start_str} --> {end_str}\n{text}\n\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def prepare_scene_segment(
        self,
        clip_path: Path,
        duration: float,
        aspect_ratio: str,
        output_path: Path
    ) -> Path:
        """
        Transcodes and normalizes an individual clip to target aspect ratio,
        30fps, YUV420p, color normalization, and exact duration.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = ASPECT_CONFIGS.get(aspect_ratio, ASPECT_CONFIGS["16:9"])
        w, h = cfg["width"], cfg["height"]

        # Color & brightness normalization filter
        # eq=brightness=0.02:contrast=1.05:saturation=1.1
        color_filter = "eq=brightness=0.02:contrast=1.05:saturation=1.08"

        if aspect_ratio in ["9:16", "1:1"]:
            # Smart Composition for non-16:9 ratios (avoids edge subject cutoff):
            # Background is blurred scaled crop, foreground is fitted aspect
            filter_complex = (
                f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
                f"boxblur=25:5,{color_filter}[bg];"
                f"[0:v]scale={w}:-1:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,fps=30,format=yuv420p"
            )
        else:
            filter_complex = (
                f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
                f"{color_filter},fps=30,format=yuv420p"
            )

        cmd = [
            FFMPEG_EXE, "-y",
            "-stream_loop", "-1", # loop if clip shorter than scene
            "-i", str(clip_path),
            "-t", str(duration),
            "-filter_complex", filter_complex,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-an", # No original audio from b-roll
            str(output_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            logger.error(f"FFmpeg segment error: {res.stderr[-300:]}")
            raise RuntimeError(f"FFmpeg failed to transcode scene segment: {res.stderr[-200:]}")
        return output_path

    def assemble_video(
        self,
        scene_segments: List[Path],
        audio_path: Path,
        aspect_ratio: str,
        output_path: Path,
        srt_path: Optional[Path] = None,
        burn_in_captions: bool = False
    ) -> Path:
        """
        Stitches pre-normalized scene segments together and multiplexes narration audio.
        Applies burned-in captions if requested.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = output_path.parent / f"tmp_concat_{aspect_ratio.replace(':', '_')}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Build concat demuxer list
        concat_list_file = temp_dir / "concat_list.txt"
        with open(concat_list_file, "w", encoding="utf-8") as f:
            for seg in scene_segments:
                # Escape single quotes and use forward slashes
                p_str = str(seg.resolve()).replace("\\", "/")
                f.write(f"file '{p_str}'\n")

        stitched_video = temp_dir / "stitched_raw.mp4"

        # Concat video streams
        cmd_concat = [
            FFMPEG_EXE, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_file),
            "-c", "copy",
            str(stitched_video)
        ]
        res = subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            logger.error(f"FFmpeg concat error: {res.stderr[-300:]}")
            raise RuntimeError("Failed to concatenate video segments")

        # Merge with audio and optional subtitles
        cmd_final = [
            FFMPEG_EXE, "-y",
            "-i", str(stitched_video),
            "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest"
        ]

        if burn_in_captions and srt_path and srt_path.exists():
            srt_str = str(srt_path.resolve()).replace("\\", "/").replace(":", "\\:")
            # High-visibility burned-in styling: font size 20, white text, semi-transparent black outline box
            subtitle_filter = f"subtitles='{srt_str}':force_style='FontSize=22,PrimaryColour=&H00FFFFFF,BorderStyle=3,Outline=2,Shadow=1,MarginV=35'"
            cmd_final.extend(["-vf", subtitle_filter, "-c:v", "libx264", "-preset", "veryfast", "-crf", "22"])
        else:
            cmd_final.extend(["-c:v", "copy"])

        cmd_final.append(str(output_path))

        res_final = subprocess.run(cmd_final, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res_final.returncode != 0:
            logger.error(f"FFmpeg final merge error: {res_final.stderr[-300:]}")
            raise RuntimeError(f"Failed to multiplex final video: {res_final.stderr[-200:]}")

        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

        return output_path

video_renderer = VideoRenderer()
