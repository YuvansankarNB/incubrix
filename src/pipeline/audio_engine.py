import os
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import imageio_ffmpeg
import edge_tts
from src.config import settings
from src.logging_config import logger

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

VOICE_MAP = {
    "en": "en-US-AriaNeural",
    "en-IN-female": "en-IN-NeerjaNeural",
    "en-IN-male": "en-IN-PrabhatNeural",
    "hi": "hi-IN-SwaraNeural",
    "hi-IN-female": "hi-IN-SwaraNeural",
    "hi-IN-male": "hi-IN-MadhurNeural",
    "ta": "ta-IN-PallaviNeural",
    "ta-IN-female": "ta-IN-PallaviNeural",
    "ta-IN-male": "ta-IN-ValluvarNeural"
}

class AudioEngineException(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False, details: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}
        super().__init__(self.message)

class AudioEngine:
    def __init__(self):
        self.whisper_model = None

    def _get_whisper_model(self):
        if self.whisper_model is None:
            from faster_whisper import WhisperModel
            logger.info("Initializing Faster-Whisper model...")
            self.whisper_model = WhisperModel(settings.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        return self.whisper_model

    async def synthesize_speech(self, text: str, voice_pref: Optional[str] = None, output_path: Optional[Path] = None) -> Path:
        if not text or not text.strip():
            raise AudioEngineException(
                code="EMPTY_SCRIPT_ERROR",
                message="The provided text script is empty.",
                retryable=False
            )
        
        voice = VOICE_MAP.get(voice_pref or "", settings.TTS_DEFAULT_VOICE_EN)
        # Check language heuristic if generic voice given
        if not voice_pref:
            # Simple script heuristic for Indic characters
            if any('\u0900' <= c <= '\u097F' for c in text): # Devanagari / Hindi
                voice = settings.TTS_DEFAULT_VOICE_HI
            elif any('\u0B80' <= c <= '\u0BFF' for c in text): # Tamil
                voice = settings.TTS_DEFAULT_VOICE_TA

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            output_path = Path(tmp.name)
            tmp.close()

        logger.info(f"Synthesizing voiceover with voice '{voice}' to {output_path}")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        return output_path

    def inspect_audio(self, audio_path: Path) -> Dict[str, Any]:
        """Validates audio integrity, duration, and checks for silence."""
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            raise AudioEngineException(
                code="INVALID_AUDIO_FORMAT",
                message="Audio file is missing or has 0 bytes.",
                retryable=False
            )

        # Probe duration and audio stats using ffmpeg
        cmd = [
            FFMPEG_EXE, "-i", str(audio_path),
            "-af", "volumedetect",
            "-f", "null", "-"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stderr = res.stderr

        if "Invalid data found when processing input" in stderr or res.returncode != 0:
            raise AudioEngineException(
                code="INVALID_AUDIO_FORMAT",
                message="Unable to decode audio format. Please provide valid mp3, wav, or m4a file.",
                retryable=False,
                details={"ffmpeg_error": stderr[-300:]}
            )

        # Parse duration
        duration = 0.0
        for line in stderr.splitlines():
            if "Duration:" in line:
                try:
                    time_str = line.split("Duration:")[1].split(",")[0].strip()
                    h, m, s = time_str.split(":")
                    duration = float(h) * 3600 + float(m) * 60 + float(s)
                except Exception:
                    pass

        if duration > settings.MAX_AUDIO_DURATION_SECONDS:
            raise AudioEngineException(
                code="AUDIO_EXCEEDS_MAX_DURATION",
                message=f"Audio duration ({duration:.1f}s) exceeds max limit of {settings.MAX_AUDIO_DURATION_SECONDS}s.",
                retryable=False,
                details={"duration": duration, "limit": settings.MAX_AUDIO_DURATION_SECONDS}
            )

        # Check for silence (mean_volume <= -60dB or max_volume <= -50dB)
        mean_vol = None
        for line in stderr.splitlines():
            if "mean_volume:" in line:
                try:
                    mean_vol = float(line.split("mean_volume:")[1].replace("dB", "").strip())
                except Exception:
                    pass

        if mean_vol is not None and mean_vol < -55.0:
            raise AudioEngineException(
                code="SILENT_AUDIO_DETECTED",
                message="The provided audio is silent or virtually inaudible.",
                retryable=False,
                details={"mean_volume_db": mean_vol}
            )

        return {"duration": duration, "mean_volume_db": mean_vol}

    def transcribe_and_align(self, audio_path: Path) -> Tuple[List[Dict[str, Any]], str, float]:
        """Extracts word-level timestamps and recognized text."""
        info = self.inspect_audio(audio_path)
        total_duration = info["duration"]

        try:
            model = self._get_whisper_model()
            segments, seg_info = model.transcribe(
                str(audio_path),
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400)
            )

            word_list = []
            full_text_parts = []

            for segment in segments:
                full_text_parts.append(segment.text.strip())
                if segment.words:
                    for w in segment.words:
                        word_list.append({
                            "word": w.word.strip(),
                            "start": round(w.start, 2),
                            "end": round(w.end, 2),
                            "probability": round(w.probability, 2)
                        })

            full_transcript = " ".join(full_text_parts).strip()

            if not full_transcript or len(word_list) == 0:
                raise AudioEngineException(
                    code="NO_SPEECH_DETECTED",
                    message="No recognizable spoken speech was found in the audio (audio may be instrumental/music only).",
                    retryable=False
                )

            return word_list, full_transcript, total_duration

        except AudioEngineException:
            raise
        except Exception as e:
            logger.error(f"Whisper alignment error: {str(e)}")
            # Fallback for offline or fast simulation if needed
            raise AudioEngineException(
                code="ALIGNMENT_FAILED",
                message=f"Failed to extract timestamps from audio: {str(e)}",
                retryable=True
            )

audio_engine = AudioEngine()
