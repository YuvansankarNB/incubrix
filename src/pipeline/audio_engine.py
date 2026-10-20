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
    "auto": None,
    "male": "en-US-GuyNeural",
    "female": "en-US-JennyNeural",
    "en": "en-US-GuyNeural",
    "en-male": "en-US-GuyNeural",
    "en-female": "en-US-JennyNeural",
    "en-US-GuyNeural": "en-US-GuyNeural",
    "en-US-ChristopherNeural": "en-US-ChristopherNeural",
    "en-US-JennyNeural": "en-US-JennyNeural",
    "en-US-AriaNeural": "en-US-AriaNeural",
    "en-US-AvaNeural": "en-US-AvaNeural",
    "en-US-EricNeural": "en-US-EricNeural",
    "en-IN-female": "en-IN-NeerjaNeural",
    "en-IN-male": "en-IN-PrabhatNeural",
    "hi": "hi-IN-MadhurNeural",
    "hi-male": "hi-IN-MadhurNeural",
    "hi-female": "hi-IN-SwaraNeural",
    "hi-IN-female": "hi-IN-SwaraNeural",
    "hi-IN-male": "hi-IN-MadhurNeural",
    "ta": "ta-IN-ValluvarNeural",
    "ta-male": "ta-IN-ValluvarNeural",
    "ta-female": "ta-IN-PallaviNeural",
    "ta-IN-female": "ta-IN-PallaviNeural",
    "ta-IN-male": "ta-IN-ValluvarNeural"
}

FEMALE_VOICES = {
    "en-us-jennyneural", "en-us-arianeural", "en-us-avaneural", "en-us-emmaneural",
    "en-gb-sonianeural", "en-in-neerjaneural", "hi-in-swaraneural", "ta-in-pallavineural"
}

MALE_VOICES = {
    "en-us-guyneural", "en-us-christopherneural", "en-us-ericneural", "en-us-andrewneural",
    "en-gb-ryanneural", "en-in-prabhatneural", "hi-in-madhurneural", "ta-in-valluvarneural"
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

    @classmethod
    def detect_speaker_gender(cls, speaker: str, context: str = "", character_type: str = "") -> str:
        """
        Accurately identifies speaker gender from character name, title, role, or context.
        Guarantees:
        - Male characters (Doctor Marcus, Farmer Henry, Businessman Alex, Presenter David, James) -> 'male'
        - Female characters (Teacher Clara, Student Emma, Dr. Sarah, Patient Elena) -> 'female'
        - Explicit gender words ('female', 'woman', 'male', 'man') have absolute priority.
        """
        import re
        s_low = (speaker or "").lower().strip()
        c_low = (context or "").lower().strip()
        ct_low = (character_type or "").lower().strip()

        s_words = set(re.findall(r'[a-zA-Z]+', s_low))
        ct_words = set(re.findall(r'[a-zA-Z]+', ct_low))
        target_words = s_words | ct_words

        female_explicit = {
            "female", "woman", "women", "girl", "lady", "ladies", "mrs", "ms", "miss",
            "madam", "she", "her", "hers", "herself", "mother", "mom", "sister",
            "daughter", "aunt", "wife", "queen", "princess", "actress", "businesswoman"
        }
        male_explicit = {
            "male", "man", "men", "boy", "guy", "gentleman", "mr", "sir", "lord",
            "he", "him", "his", "himself", "father", "dad", "brother", "son",
            "uncle", "husband", "king", "prince", "actor", "businessman"
        }

        # 1. Direct explicit keyword override in speaker or character_type
        has_fem_exp = bool(target_words & female_explicit)
        has_mal_exp = bool(target_words & male_explicit)

        if has_fem_exp and not has_mal_exp:
            return "female"
        if has_mal_exp and not has_fem_exp:
            return "male"

        # 2. Check 3D Character Model definition from avatar_service
        try:
            from src.pipeline.avatar_service import avatar_service
            target_role = s_low if (s_low and s_low not in ["speaker", "narrator"]) else (ct_low or s_low or "narrator")
            char_gender = avatar_service.get_character_gender(target_role)
            if char_gender in ["male", "female"]:
                return char_gender
        except Exception:
            pass

        # 3. Specific known character names & roles (matched as whole words)
        female_names_and_roles = {
            "sarah", "emily", "elena", "clara", "emma", "jenny", "jennifer", "anna",
            "maria", "linda", "rachel", "laura", "neerja", "swara", "pallavi", "priya",
            "teacher", "student", "nurse"
        }
        male_names_and_roles = {
            "doctor", "physician", "surgeon", "marcus", "alex", "james",
            "david", "henry", "john", "michael", "robert",
            "william", "thomas", "guy", "christopher", "eric", "andrew", "ryan",
            "prabhat", "madhur", "valluvar",
            "farmer", "executive", "businessman", "business", "narrator", "presenter",
            "anchor", "engineer"
        }

        if target_words & female_names_and_roles:
            return "female"
        if target_words & male_names_and_roles:
            return "male"

        # Context-based default: 'patient' paired with 'doctor'
        if "patient" in target_words:
            return "female"
        if "doctor" in target_words or "physician" in target_words:
            return "male"

        # Check pronouns in context text
        c_words = set(re.findall(r'[a-zA-Z]+', c_low))
        if c_words & {"she", "her", "hers"}:
            return "female"
        if c_words & {"he", "him", "his"}:
            return "male"

        return "male"

    def resolve_voice_for_speaker(self, speaker: str, gender: str, language: str = "en") -> str:
        """
        Assigns an expressive, natural-sounding voice strictly matching the character's gender and language.
        Male character -> ALWAYS male voice (Christopher, Guy, Prabhat, Madhur, Valluvar).
        Female character -> ALWAYS female voice (Jenny, Aria, Ava, Neerja, Swara, Pallavi).
        """
        gender = (gender or "").lower().strip()
        if gender not in ["male", "female"]:
            gender = self.detect_speaker_gender(speaker)

        if language == "hi":
            return "hi-IN-SwaraNeural" if gender == "female" else "hi-IN-MadhurNeural"
        elif language == "ta":
            return "ta-IN-PallaviNeural" if gender == "female" else "ta-IN-ValluvarNeural"
        elif language == "en-IN":
            return "en-IN-NeerjaNeural" if gender == "female" else "en-IN-PrabhatNeural"
        else:
            # English (en-US)
            s_low = (speaker or "").lower()
            if gender == "female":
                if any(w in s_low for w in ["student", "emma", "young", "girl"]):
                    return "en-US-AvaNeural"
                if any(w in s_low for w in ["doctor", "physician", "sarah", "emily", "clinical"]):
                    return "en-US-AriaNeural"
                if any(w in s_low for w in ["teacher", "professor", "clara", "educator"]):
                    return "en-US-JennyNeural"
                return "en-US-JennyNeural"
            else:
                # Male
                if any(w in s_low for w in ["doctor", "physician", "surgeon", "executive", "director", "marcus"]):
                    return "en-US-ChristopherNeural"
                if any(w in s_low for w in ["farmer", "henry", "grower", "soil"]):
                    return "en-US-GuyNeural"
                if any(w in s_low for w in ["student", "boy", "young"]):
                    return "en-US-EricNeural"
                return "en-US-GuyNeural"

    async def _synthesize_multi_speaker_dialogue(
        self,
        turns: List[Dict[str, Any]],
        voice_pref: Optional[str],
        output_path: Path
    ):
        """
        Synthesizes each character's dialogue turn with a distinct, natural voice,
        matching gender (male voice for male characters, female voice for female characters),
        and strictly maintaining voice consistency across all scenes.
        """
        speakers = []
        for t in turns:
            spk = t["speaker"].strip()
            if spk.lower() not in [s.lower() for s in speakers]:
                speakers.append(spk)

        full_text = " ".join(t["text"] for t in turns)
        is_hindi = any('\u0900' <= c <= '\u097F' for c in full_text) or "hi" in (voice_pref or "")
        is_tamil = any('\u0B80' <= c <= '\u0BFF' for c in full_text) or "ta" in (voice_pref or "")
        lang = "hi" if is_hindi else ("ta" if is_tamil else ("en-IN" if "en-IN" in (voice_pref or "") else "en"))

        # Maintain or build persistent session voice map
        if not hasattr(self, "speaker_voice_map") or self.speaker_voice_map is None:
            self.speaker_voice_map = {}

        for idx, spk in enumerate(speakers):
            spk_key = spk.lower().strip()
            if spk_key not in self.speaker_voice_map:
                gender = self.detect_speaker_gender(spk, full_text)
                # If two speakers and first is male, second defaults to female if ambiguous
                if len(speakers) == 2 and idx == 1:
                    first_gender = self.detect_speaker_gender(speakers[0], full_text)
                    if first_gender == "male" and gender == "male" and "doctor" not in spk_key and "marcus" not in spk_key:
                        gender = "female"
                    elif first_gender == "female" and gender == "female" and "teacher" not in spk_key and "sarah" not in spk_key:
                        gender = "male"
                voice = self.resolve_voice_for_speaker(spk, gender, lang)
                self.speaker_voice_map[spk_key] = {"voice": voice, "gender": gender}
                logger.info(f"Assigned character '{spk}' ({gender}) -> voice '{voice}'")

        temp_files = []
        turn_segments = []
        current_time = 0.0

        try:
            for idx, turn in enumerate(turns):
                t_spk = turn["speaker"].strip()
                t_text = turn["text"].strip()
                spk_info = self.speaker_voice_map.get(t_spk.lower().strip(), {"voice": "en-US-GuyNeural", "gender": "male"})
                t_voice = spk_info["voice"]
                t_gender = spk_info["gender"]

                turn["voice"] = t_voice
                turn["gender"] = t_gender

                t_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                t_path = Path(t_mp3.name)
                t_mp3.close()
                temp_files.append(t_path)

                logger.info(f"Synthesizing turn {idx} ({t_spk}, {t_gender}) with voice '{t_voice}': '{t_text[:35]}...'")
                try:
                    comm = edge_tts.Communicate(t_text, t_voice)
                    await comm.save(str(t_path))
                except Exception as e:
                    logger.warning(f"Voice synthesis failed for '{t_voice}': {e}. Falling back to default verified voice.")
                    fallback_voice = "en-US-JennyNeural" if t_gender == "female" else "en-US-GuyNeural"
                    turn["voice_fallback"] = True
                    comm = edge_tts.Communicate(t_text, fallback_voice)
                    await comm.save(str(t_path))
                    fallback_voice = settings.TTS_DEFAULT_VOICE_EN
                    turn["voice_fallback"] = True
                    comm = edge_tts.Communicate(t_text, fallback_voice)
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

    async def synthesize_speech(
        self,
        text: str,
        voice_pref: Optional[str] = None,
        output_path: Optional[Path] = None,
        character_type: Optional[str] = None,
        voice_gender: Optional[str] = None
    ) -> Path:
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
                logger.info(f"Detected {len(turns)} dialogue turns. Synthesizing multi-speaker audio with character-matched voices.")
                await self._synthesize_multi_speaker_dialogue(turns, voice_pref, output_path)
                return output_path
            except Exception as e:
                logger.warning(f"Multi-speaker dialogue synthesis failed: {e}. Falling back to single-pass.")

        # Single-pass: ALWAYS strip speaker prefixes so character names like 'Doctor:' are NEVER spoken
        clean_text = self.strip_speaker_prefixes(text)

        # 1. Determine target character gender
        target_gender = None
        if voice_gender and voice_gender.lower() in ["male", "female"]:
            target_gender = voice_gender.lower()
        elif character_type:
            target_gender = self.detect_speaker_gender("", clean_text, character_type=character_type)
        else:
            target_gender = self.detect_speaker_gender("", clean_text)

        # 2. Determine target language
        is_hindi = any('\u0900' <= c <= '\u097F' for c in clean_text) or (voice_pref and "hi" in voice_pref)
        is_tamil = any('\u0B80' <= c <= '\u0BFF' for c in clean_text) or (voice_pref and "ta" in voice_pref)
        lang = "hi" if is_hindi else ("ta" if is_tamil else ("en-IN" if (voice_pref and "en-in" in voice_pref.lower()) else "en"))

        # 3. Resolve voice ensuring strict character-gender alignment
        voice = None
        if voice_pref and voice_pref != "auto":
            candidate = VOICE_MAP.get(voice_pref, voice_pref)
            cand_l = candidate.lower()
            cand_is_fem = any(fv in cand_l for fv in ["jenny", "aria", "ava", "emma", "swara", "pallavi", "neerja", "female"])
            cand_is_mal = any(mv in cand_l for mv in ["guy", "christopher", "eric", "andrew", "madhur", "valluvar", "prabhat", "male"])

            # Enforce gender alignment: Never play female voice for male character or vice-versa
            if target_gender == "male" and cand_is_fem:
                logger.info(f"Corrected requested voice '{candidate}' to natural male voice to match male character ({character_type or 'male'}).")
                voice = self.resolve_voice_for_speaker(character_type or "narrator", "male", lang)
            elif target_gender == "female" and cand_is_mal:
                logger.info(f"Corrected requested voice '{candidate}' to natural female voice to match female character ({character_type or 'female'}).")
                voice = self.resolve_voice_for_speaker(character_type or "teacher", "female", lang)
            else:
                voice = candidate

        if not voice:
            role_hint = character_type or "narrator"
            voice = self.resolve_voice_for_speaker(role_hint, target_gender, lang)

        logger.info(f"Synthesizing voiceover for character '{character_type}' ({target_gender}) with voice '{voice}' to {output_path} (prefixes stripped)")
        try:
            communicate = edge_tts.Communicate(clean_text, voice)
            await communicate.save(str(output_path))
        except Exception as e:
            logger.warning(f"Voice synthesis failed for '{voice}': {e}. Falling back to default verified voice.")
            fallback = "en-US-JennyNeural" if target_gender == "female" else "en-US-GuyNeural"
            communicate = edge_tts.Communicate(clean_text, fallback)
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
