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

    @staticmethod
    def strip_speaker_prefixes(raw: str) -> str:
        """
        Strips leading speaker labels (e.g. 'Doctor: ...', 'Patient: ...') from narration text
        so that speaker names are never spoken in the voiceover.
        """
        import re
        pattern = re.compile(r'^\s*(?:\[|\()?([A-Za-z0-9\.\s_]{2,25}?)(?:\]|\))?\s*:\s*(.+)$', re.MULTILINE)
        return pattern.sub(r'\2', raw).strip()

    @staticmethod
    def parse_dialogue_turns(raw: str) -> List[Dict[str, Any]]:
        """
        Parses multi-character dialogue scripts into turns with speaker tags and clean spoken text.
        """
        import re
        pattern = re.compile(r'^\s*(?:\[|\()?([A-Za-z0-9\.\s_]{2,25}?)(?:\]|\))?\s*:\s*(.+)$')
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        turns = []
        for line in lines:
            m = pattern.match(line)
            if m:
                turns.append({"speaker": m.group(1).strip(), "text": m.group(2).strip()})
        return turns

    async def _synthesize_multi_speaker_dialogue(
        self,
        turns: List[Dict[str, Any]],
        voice_pref: Optional[str],
        output_path: Path
    ):
        """
        Synthesizes each character's dialogue turn with a distinct, natural voice,
        completely omitting character name prefixes from the spoken audio.
        """
        speakers = []
        for t in turns:
            spk = t["speaker"].lower()
            if spk not in [s.lower() for s in speakers]:
                speakers.append(t["speaker"])

        full_text = " ".join(t["text"] for t in turns)
        is_hindi = any('\u0900' <= c <= '\u097F' for c in full_text) or "hi" in (voice_pref or "")
        is_tamil = any('\u0B80' <= c <= '\u0BFF' for c in full_text) or "ta" in (voice_pref or "")

        speaker_voices = {}
        for idx, spk in enumerate(speakers):
            s_low = spk.lower()
            if is_hindi:
                speaker_voices[spk] = "hi-IN-MadhurNeural" if idx % 2 == 0 else "hi-IN-SwaraNeural"
            elif is_tamil:
                speaker_voices[spk] = "ta-IN-ValluvarNeural" if idx % 2 == 0 else "ta-IN-PallaviNeural"
            else:
                if any(w in s_low for w in ["doctor", "physician", "surgeon", "alex", "king"]):
                    speaker_voices[spk] = "en-US-ChristopherNeural"
                elif any(w in s_low for w in ["patient", "nova", "morgan", "advisor"]):
                    speaker_voices[spk] = "en-US-JennyNeural"
                else:
                    speaker_voices[spk] = "en-US-ChristopherNeural" if idx % 2 == 0 else "en-US-JennyNeural"

        temp_files = []
        turn_segments = []
        current_time = 0.0

        try:
            for idx, turn in enumerate(turns):
                t_spk = turn["speaker"]
                t_text = turn["text"].strip()
                t_voice = speaker_voices.get(t_spk, settings.TTS_DEFAULT_VOICE_EN)

                t_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                t_path = Path(t_mp3.name)
                t_mp3.close()
                temp_files.append(t_path)

                logger.info(f"Synthesizing turn {idx} ({t_spk}) with voice '{t_voice}': '{t_text[:35]}...'")
                comm = edge_tts.Communicate(t_text, t_voice)
                await comm.save(str(t_path))

                info = self.inspect_audio(t_path)
                turn_dur = round(info["duration"], 2)
                turn["start_time"] = round(current_time, 2)
                turn["end_time"] = round(current_time + turn_dur, 2)
                turn["duration"] = turn_dur
                current_time += turn_dur

                turn_segments.append(t_path)

            # Concatenate turn audio files using ffmpeg concat demuxer
            concat_list = tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8")
            for seg in turn_segments:
                seg_esc = str(seg).replace('\\', '/')
                concat_list.write(f"file '{seg_esc}'\n")
            concat_list.close()
            temp_files.append(Path(concat_list.name))

            cmd = [
                FFMPEG_EXE, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list.name,
                "-c", "copy",
                str(output_path)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg concat failed: {res.stderr}")

        finally:
            for tf in temp_files:
                try:
                    if tf.exists():
                        tf.unlink()
                except Exception:
                    pass

    async def synthesize_speech(self, text: str, voice_pref: Optional[str] = None, output_path: Optional[Path] = None) -> Path:
        if not text or not text.strip():
            raise AudioEngineException(
                code="EMPTY_SCRIPT_ERROR",
                message="The provided text script is empty.",
                retryable=False
            )

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            output_path = Path(tmp.name)
            tmp.close()

        # Check for dialogue turns with speaker prefixes (e.g. Doctor: ... Patient: ...)
        turns = self.parse_dialogue_turns(text)
        self.last_dialogue_turns = turns

        if len(turns) >= 2:
            try:
                logger.info(f"Detected {len(turns)} dialogue turns. Synthesizing multi-speaker audio without speaker prefixes.")
                await self._synthesize_multi_speaker_dialogue(turns, voice_pref, output_path)
                return output_path
            except Exception as e:
                logger.warning(f"Multi-speaker dialogue synthesis failed: {e}. Falling back to single-pass.")

        # Single-pass: ALWAYS strip speaker prefixes so character names like 'Doctor:' are NEVER spoken
        clean_text = self.strip_speaker_prefixes(text)

        voice = VOICE_MAP.get(voice_pref or "", settings.TTS_DEFAULT_VOICE_EN)
        if not voice_pref:
            if any('\u0900' <= c <= '\u097F' for c in clean_text): # Devanagari / Hindi
                voice = settings.TTS_DEFAULT_VOICE_HI
            elif any('\u0B80' <= c <= '\u0BFF' for c in clean_text): # Tamil
                voice = settings.TTS_DEFAULT_VOICE_TA

        logger.info(f"Synthesizing voiceover with voice '{voice}' to {output_path} (prefixes stripped)")
        communicate = edge_tts.Communicate(clean_text, voice)
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
