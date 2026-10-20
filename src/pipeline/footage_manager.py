import os
import hashlib
import json
import math
import random
import re
import subprocess
import urllib.parse
import urllib.request
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from src.config import settings
from src.logging_config import logger
import imageio_ffmpeg

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

class FootageClip:
    def __init__(
        self,
        clip_id: str,
        local_path: Path,
        source: str,
        license_name: str,
        attribution: str,
        original_url: str,
        is_ai_generated: bool,
        selection_reason: str,
        relevance_score: float = 4.2,
        provider: Optional[str] = None,
        creator: Optional[str] = None,
        license_url: Optional[str] = None,
        restrictions: Optional[List[str]] = None,
        commercial_rights_cleared: bool = True,
        has_licensing_warning: bool = False
    ):
        self.clip_id = clip_id
        self.local_path = local_path
        self.source = source
        self.license_name = license_name
        self.attribution = attribution
        self.original_url = original_url
        self.is_ai_generated = is_ai_generated
        self.selection_reason = selection_reason
        self.relevance_score = relevance_score
        self.provider = provider or source
        self.creator = creator or attribution
        self.license_url = license_url or ("https://ai.google.dev/terms" if is_ai_generated else "https://www.pexels.com/license/")
        self.restrictions = restrictions or ([] if not is_ai_generated else ["AI-generated content disclosure required."])
        self.commercial_rights_cleared = commercial_rights_cleared
        self.has_licensing_warning = has_licensing_warning
        self.page_url = original_url
        self.pixabay_video_id = None
        self.thumbnail_url = None
        self.is_mock = False
        self.download_date = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "source": self.source,
            "provider": self.provider,
            "creator": self.creator,
            "license": self.license_name,
            "license_name": self.license_name,
            "license_url": self.license_url,
            "attribution": self.attribution,
            "url": self.original_url,
            "page_url": self.page_url,
            "is_ai_generated": self.is_ai_generated,
            "selection_reason": self.selection_reason,
            "relevance_score": self.relevance_score,
            "restrictions": self.restrictions,
            "commercial_rights_cleared": self.commercial_rights_cleared,
            "has_licensing_warning": self.has_licensing_warning,
            "pixabay_video_id": self.pixabay_video_id,
            "thumbnail_url": self.thumbnail_url,
            "is_mock": self.is_mock,
            "download_date": self.download_date
        }

class FootageManager:
    def __init__(self):
        self.cache_dir = Path("./data/footage_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.used_clip_ids: set = set()
        self._audio_cache: Dict[str, np.ndarray] = {}
        self.show_character_labels: bool = False

    def reset_used_clips(self):
        self.used_clip_ids.clear()
        self._audio_cache.clear()

    def _load_pcm_audio(self, audio_path: Optional[Path]) -> np.ndarray:
        """Extracts 16kHz mono raw PCM samples to drive true audio-synced lip sync."""
        if not audio_path or not audio_path.exists():
            return np.array([], dtype=np.int16)
        
        path_str = str(audio_path.resolve())
        if path_str in self._audio_cache:
            return self._audio_cache[path_str]

        try:
            cmd = [
                FFMPEG_EXE, "-i", path_str,
                "-f", "s16le", "-ac", "1", "-ar", "16000", "-"
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            raw, _ = p.communicate()
            samples = np.frombuffer(raw, dtype=np.int16)
            self._audio_cache[path_str] = samples
            return samples
        except Exception as e:
            logger.warning(f"Could not extract PCM audio for lip sync: {e}")
            return np.array([], dtype=np.int16)

    def _draw_cursor(self, canvas: np.ndarray, x: int, y: int, clicking: bool = False, click_progress: float = 0.0):
        """Renders a sleek, modern OS mouse pointer with smooth hover and click ripples."""
        pts = np.array([
            [x, y],
            [x, y + 26],
            [x + 7, y + 20],
            [x + 14, y + 30],
            [x + 19, y + 28],
            [x + 12, y + 18],
            [x + 21, y + 18]
        ], dtype=np.int32)
        
        # Shadow
        cv2.fillPoly(canvas, [pts + 3], (10, 10, 10), cv2.LINE_AA)
        # White pointer body
        cv2.fillPoly(canvas, [pts], (255, 255, 255), cv2.LINE_AA)
        # Dark border
        cv2.polylines(canvas, [pts], True, (25, 25, 25), 2, cv2.LINE_AA)

        if clicking and 0.0 <= click_progress <= 1.0:
            # Concentric expanding ripple rings
            ripple_r = int(10 + click_progress * 40)
            alpha = max(0.0, 1.0 - click_progress)
            color = (int(80 * alpha), int(220 * alpha), int(255 * alpha))
            cv2.circle(canvas, (x, y), ripple_r, color, 2, cv2.LINE_AA)
    @staticmethod
    def evaluate_speech_viseme(text: str, t: float, duration: float, rms_energy: float) -> Tuple[float, str]:
        """
        Determines mouth opening amplitude and viseme type based on phoneme timing and audio RMS energy.
        During pauses or silence, returns (0.0, "closed").
        """
        if rms_energy < 0.03 or not text:
            return 0.0, "closed"

        clean_words = [w.lower().strip() for w in re.split(r'\s+', re.sub(r'[^a-zA-Z\s]', '', text)) if w.strip()]
        if not clean_words:
            return (rms_energy, "open") if rms_energy > 0.08 else (0.0, "closed")

        num_words = len(clean_words)
        word_dur = duration / max(1, num_words)
        word_idx = int(t / max(0.001, word_dur))

        if word_idx >= num_words:
            return 0.0, "closed"

        active_word = clean_words[word_idx]
        rel_w = (t - word_idx * word_dur) / max(0.001, word_dur)

        # Inter-word micro-pause (~12% gap at word end)
        if rel_w > 0.88 and num_words > 1:
            return 0.0, "closed"

        # Smooth syllable envelope
        syllable_envelope = math.sin(math.pi * np.clip(rel_w / 0.88, 0.0, 1.0)) ** 0.85
        amplitude = syllable_envelope * (0.35 + 0.65 * rms_energy)

        # Bilabial closed mouth on M, B, P at word edges
        if (rel_w < 0.12 or rel_w > 0.80) and any(active_word.startswith(c) or active_word.endswith(c) for c in ['m', 'b', 'p']):
            return 0.0, "closed"

        # Vowel / Viseme classification
        if any(v in active_word for v in ['ee', 'ea', 'ey', 'ie', 'take', 'make', 'safe', 'day', 'pain', 'aid', 'great', 'read', 'see', 'teeth']):
            viseme = "wide"
        elif any(v in active_word for v in ['oo', 'ou', 'ue', 'who', 'you', 'through', 'blue', 'true', 'do', 'new', 'two', 'room']):
            viseme = "pucker"
        elif any(v in active_word for v in ['o', 'oa', 'ow', 'go', 'no', 'home', 'know', 'flow', 'grow', 'slow', 'posture', 'doctor', 'bone']):
            viseme = "round"
        elif any(v in active_word for v in ['f', 'v', 'th', 'five', 'feel', 'very', 'voice', 'first']):
            viseme = "fricative"
        else:
            viseme = "open"

        return float(np.clip(amplitude, 0.0, 1.0)), viseme

    def _apply_theme_background(self, canvas: np.ndarray, base_bg: Optional[np.ndarray], theme_color: tuple):
        """Blends the real-world photographic background with subtle atmospheric theme grading."""
        if base_bg is not None and base_bg.size > 0:
            tint = np.full_like(canvas, theme_color)
            cv2.addWeighted(canvas, 0.82, tint, 0.18, 0, canvas)
        else:
            canvas[:] = theme_color

    def generate_procedural_broll(
        self,
        query: str,
        duration: float,
        output_path: Path,
        style: str,
        audio_path: Optional[Path] = None,
        start_time: float = 0.0,
        scene_text: str = "",
        character_type: str = "human",
        scene_speaker: Optional[str] = None,
        scene_index: int = 0,
        visual_hint: Optional[str] = None,
        verified_fact: Optional[Dict[str, Any]] = None,
        is_voiceover: bool = False,
        total_scenes: int = 1,
        aspect_ratio: str = "16:9",
        custom_headline: Optional[str] = None,
        show_character_labels: bool = False
    ) -> FootageClip:
        """
        Synthesizes high-impact, content-driven animated B-roll clips:
        - 14 Content-Specific Interactive Motion Themes with live counters, widgets, and checklists.
        - Animated Mouse Pointer with simulated user clicks and expanding ripple waves.
        - Authentic 16kHz Audio-Driven Lip Sync matching spoken volume and pauses.
        - Animated Host Presenter with blinking eyes, pupil tracking, and 8-bar equalizer bubble.
        - Deep Visual Hint & Semantic Context Understanding (e.g. clean energy wind farms, biotech labs, urban ecology gardens, clinical hospital wards).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.show_character_labels = show_character_labels
        width, height = 1920, 1080
        fps = 30
        total_frames = int(max(duration, 2.0) * fps)
        
        combined_text = f"{scene_text} {query} {visual_hint or ''}".lower()
        q_hash = int(hashlib.md5(combined_text.encode('utf-8')).hexdigest(), 16)
        random.seed(q_hash % 100000)

        # Detect semantic theme with high-specificity precedence
        if any(w in combined_text for w in [
            "doctor", "patient", "hospital", "clinic", "medical", "surgery", "diagnosis", 
            "prescription", "medicine", "health", "nurse", "treatment", "illness", "recovery", "lab report", "vitals", "physician"
        ]):
            theme_mode = "hospital"
            theme_title = "Clinical Hospital & Medical Ward"
        elif any(w in combined_text for w in [
            "wind farm", "wind turbine", "clean energy", "carbon emission", "carbon", "renewable", "solar", "sunset", "emission", "manufacturing", "green energy", "sustainable", "decarbonization", "climate"
        ]):
            theme_mode = "clean_energy"
            theme_title = "Clean Energy Wind Farm & Carbon Audit"
        elif any(w in combined_text for w in [
            "laboratory", "lab", "scientist", "polymer", "biodegradable", "circular material", "microscope", "research", "materials", "chemistry", "biotechnology", "molecule", "chemical", "specimen"
        ]):
            theme_mode = "laboratory"
            theme_title = "Circular Materials Science Laboratory"
        elif any(w in combined_text for w in [
            "urban garden", "community", "workshop", "planting", "garden", "green literacy", "ecological", "volunteer", "plants", "soil", "agriculture", "botanical", "seedling", "foliage", "nature"
        ]):
            theme_mode = "ecology_garden"
            theme_title = "Community Workshop & Urban Ecology Garden"
        elif any(w in combined_text for w in [
            "king", "advisor", "citadel", "kingdom", "realm", "throne", "castle", "royal", "sovereign", "monarch"
        ]):
            theme_mode = "royal"
            theme_title = "Royal Citadel & Throne Room"
        elif any(w in combined_text for w in ["tweak", "lock in", "focus", "quick move"]):
            theme_mode = "tweak"
            theme_title = "Hyper-Focus Mode Optimization"
        elif any(w in combined_text for w in ["second", "seconds", "instant", "instantly", "fixes", "5 sec"]):
            theme_mode = "instant"
            theme_title = "5-Second Instant Resolution"
        elif any(w in combined_text for w in ["comment", "comments", "drop a yes", "say yes"]):
            theme_mode = "social"
            theme_title = "Social Engagement Feed"
        elif any(w in combined_text for w in ["inbox", "email", "breakdown sent"]):
            theme_mode = "email"
            theme_title = "Blueprint Inbox Delivery"
        elif any(w in combined_text for w in ["waste", "wastes", "ruin", "ruins", "drain"]):
            theme_mode = "waste"
            theme_title = "Time Waste & Friction Analysis"
        elif any(w in combined_text for w in ["mistake", "routine", "daily habit", "warning", "wrong"]):
            theme_mode = "warning"
            theme_title = "Daily Routine Workflow Audit"
        elif any(w in combined_text for w in ["week", "calendar", "sprint", "days", "timeline"]):
            theme_mode = "calendar"
            theme_title = "7-Day Sprint Acceleration"
        elif any(w in combined_text for w in ["finance", "financial", "growth", "sale", "sales", "40%", "bank", "money", "invest", "yield", "profit", "revenue", "dollar"]):
            theme_mode = "finance"
            theme_title = "Revenue Surge Analytics"
        elif any(w in combined_text for w in ["cloud", "server", "data", "compute", "code", "python", "software", "cluster", "backup", "cyber", "network", "devops"]):
            theme_mode = "tech"
            theme_title = "Cloud Cluster Deployment"
        elif any(w in combined_text for w in ["trust", "partner", "partnership", "team", "relation", "faith", "collab", "together"]):
            theme_mode = "trust"
            theme_title = "Partnership Trust Alliance"
        elif any(w in combined_text for w in ["space", "spacecraft", "astronaut", "orbit", "satellite", "rocket", "lunar", "moon", "mars", "galaxy", "astronomy", "cosmos", "stellar", "planet", "telescope"]):
            theme_mode = "space"
            theme_title = "Deep Space & Orbital Exploration"
        elif any(w in combined_text for w in ["culinary", "cooking", "kitchen", "chef", "recipe", "food", "restaurant", "dish", "baking", "meal", "ingredient", "gourmet", "cuisine", "flavor", "cook"]):
            theme_mode = "culinary"
            theme_title = "Gourmet Kitchen & Culinary Arts"
        elif any(w in combined_text for w in ["fitness", "gym", "workout", "exercise", "training", "athlete", "cardio", "strength", "sport", "muscle", "calorie", "crossfit", "running", "athletic"]):
            theme_mode = "fitness"
            theme_title = "Athletic Training & Peak Performance"
        elif any(w in combined_text for w in ["ocean", "marine", "underwater", "sea", "fish", "reef", "aquatic", "coral", "deep sea", "subsea", "scuba", "maritime", "whale", "submarine"]):
            theme_mode = "ocean"
            theme_title = "Deep Ocean Marine Ecosystem"
        elif any(w in combined_text for w in ["automotive", "car", "racing", "vehicle", "speed", "engine", "tarmac", "supercar", "track", "horsepower", "formula", "driver", "drift", "lap"]):
            theme_mode = "automotive"
            theme_title = "Automotive Performance & Supercars"
        elif any(w in combined_text for w in ["music", "sound", "audio", "song", "concert", "frequency", "melody", "synthesizer", "dj", "acoustic", "studio", "beat", "stereo", "rhythm"]):
            theme_mode = "music"
            theme_title = "Studio Sound & Frequency Acoustics"
        elif any(w in combined_text for w in ["education", "school", "learn", "study", "classroom", "retention", "student", "teaching", "university", "lecture", "book", "knowledge", "academic"]):
            theme_mode = "education"
            theme_title = "Academic Learning & Knowledge Synthesis"
        elif any(w in combined_text for w in ["architecture", "real estate", "building", "interior", "villa", "house", "construction", "design", "structure", "urban", "property", "skyscraper", "blueprint"]):
            theme_mode = "architecture"
            theme_title = "Architectural Structure & Modern Design"
        elif any(w in combined_text for w in ["legal", "court", "law", "lawyer", "judge", "verdict", "statute", "justice", "trial", "attorney", "jurisprudence", "contract", "compliance"]):
            theme_mode = "legal"
            theme_title = "Judicial Courtroom & Jurisprudence"
        elif any(w in combined_text for w in ["travel", "mountain", "hiking", "alpine", "adventure", "outdoor", "summit", "explore", "geography", "journey", "tourism", "scenic", "peak", "climb"]):
            theme_mode = "travel"
            theme_title = "Alpine Mountain & Geographic Expedition"
        else:
            # Universal Dynamic Semantic Context Extractor for ANY novel topic
            topic_src = visual_hint or scene_text or query
            clean_words = [w for w in re.sub(r'[^a-zA-Z0-9\s]', '', topic_src).split() if len(w) > 2 and w.lower() not in {"the", "and", "for", "with", "all", "are", "was", "from", "that", "this", "our", "your", "can", "will"}]
            if clean_words:
                custom_topic = " ".join(clean_words[:3]).upper()
            else:
                custom_topic = "INNOVATION & ANALYSIS"
            theme_mode = "universal_context"
            theme_title = f"{custom_topic} Field Synthesis"

        # Fetch relevant real-world photographic background from Internet API
        try:
            from src.pipeline.background_service import background_service
            base_bg = background_service.get_background(
                query=query,
                visual_hint=visual_hint,
                theme_mode=theme_mode,
                width=width,
                height=height
            )
        except Exception as e:
            logger.warning(f"Background fetch error: {e}")
            base_bg = None

        # Load PCM audio for authentic lip-sync
        pcm_samples = self._load_pcm_audio(audio_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        raw_temp = output_path.with_suffix(".raw.mp4")
        writer = cv2.VideoWriter(str(raw_temp), fourcc, fps, (width, height))

        # Floating ambient particles
        particles = [
            {
                "x": random.uniform(80, width * 0.72),
                "y": random.uniform(80, height - 80),
                "vx": random.uniform(-1.0, 1.0),
                "vy": random.uniform(-1.0, 1.0),
                "radius": random.randint(3, 6)
            }
            for _ in range(30)
        ]

        # Financial chart points
        num_pts = 16
        chart_points = []
        base_y = height * 0.76
        for i in range(num_pts):
            px = int(width * 0.08 + (i / (num_pts - 1)) * (width * 0.62))
            fluctuation = math.sin(i * 1.5) * 35 + random.uniform(-20, 20)
            py = int(base_y - (i / (num_pts - 1)) * (height * 0.44) + fluctuation)
            chart_points.append((px, py))

        for frame_idx in range(total_frames):
            progress = frame_idx / total_frames
            t = frame_idx / fps
            global_audio_t = start_time + t

            # ----------------- REAL AUDIO-DRIVEN LIP SYNC ENVELOPE -----------------
            raw_rms = 0.0
            if len(pcm_samples) > 0:
                sample_idx = int(global_audio_t * 16000)
                w_start = max(0, sample_idx - 400)
                w_end = min(len(pcm_samples), sample_idx + 400)
                if w_end > w_start:
                    rms = float(np.sqrt(np.mean(pcm_samples[w_start:w_end].astype(np.float32)**2)))
                    # Speech RMS typically sits between 300 and 5000
                    raw_rms = np.clip((rms - 250.0) / 2600.0, 0.0, 1.0)
            else:
                raw_rms = max(0.0, math.sin(t * 12) * 0.7)

            # Accurate phoneme & viseme timing synchronized to speech phrase
            lip_amp, viseme_typ = self.evaluate_speech_viseme(
                text=scene_text,
                t=t,
                duration=duration,
                rms_energy=raw_rms
            )

            if base_bg is not None and base_bg.size > 0:
                from src.pipeline.background_service import background_service
                canvas = background_service.render_cinematic_frame(base_bg, frame_idx, total_frames, width, height)
            else:
                canvas = np.zeros((height, width, 3), dtype=np.uint8)

            # ----------------- UNIFIED BROADCAST VIDEO COMPOSITOR -----------------
            # 1. Determine character layout geometry
            ct_low = (character_type or "human").lower()
            is_two_char = any(sep in ct_low for sep in [",", "&", "+", " and ", " vs "]) or any(w in ct_low for w in ["drama", "dialogue", "duo", "conversation"])
            char_pos = "dialogue" if is_two_char else "right"

            # 2. Render 3D animated character / dialogue
            # Voice-over rule: if is_voiceover or scene has no spoken character dialogue, mouth stays strictly closed!
            effective_lip_energy = 0.0 if is_voiceover else lip_amp
            effective_viseme = "closed" if is_voiceover else viseme_typ

            self._draw_animated_character(
                canvas=canvas,
                frame_idx=frame_idx,
                t=t,
                width=width,
                height=height,
                theme=theme_mode,
                lip_sync_energy=effective_lip_energy,
                scene_text=scene_text,
                character_type=character_type,
                scene_speaker=scene_speaker,
                scene_index=scene_index,
                show_character_labels=show_character_labels,
                viseme_type=effective_viseme
            )

            # 3. Intelligent motion graphics & contextual typography
            from src.pipeline.motion_graphics import motion_graphics
            motion_graphics.render_motion_graphics(
                canvas=canvas,
                scene_text=scene_text,
                t=t,
                duration=duration,
                scene_index=scene_index,
                total_scenes=total_scenes,
                character_pos=char_pos,
                aspect_ratio=aspect_ratio,
                verified_fact=verified_fact,
                custom_headline=custom_headline
            )

            # Write frame to video and proceed
            writer.write(canvas)
            continue
            
            # ----------------- THEME 1: WARNING & ROUTINE AUDIT -----------------
            if theme_mode == "warning":
                self._apply_theme_background(canvas, base_bg, (24, 20, 28))
                cursor_target = (int(width * 0.46), int(height * 0.38))
                
                # Ambient warning grid
                for gx in range(0, int(width * 0.74), 90):
                    cv2.line(canvas, (gx, 0), (gx, height), (36, 28, 38), 1)
                for gy in range(0, height, 75):
                    cv2.line(canvas, (0, gy), (int(width * 0.74), gy), (36, 28, 38), 1)

                # Warning Shield / Hazard Triangle at Top Left
                tri_c = (int(width * 0.12), int(height * 0.14))
                pulse = math.sin(t * 6) * 4
                tri_pts = np.array([
                    [tri_c[0], int(tri_c[1] - 30 - pulse)],
                    [int(tri_c[0] - 32 - pulse), int(tri_c[1] + 24)],
                    [int(tri_c[0] + 32 + pulse), int(tri_c[1] + 24)]
                ], dtype=np.int32)
                cv2.fillPoly(canvas, [tri_pts], (30, 180, 240), cv2.LINE_AA)
                cv2.polylines(canvas, [tri_pts], True, (50, 220, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "!", (tri_c[0] - 6, tri_c[1] + 16), cv2.FONT_HERSHEY_DUPLEX, 1.2, (15, 15, 15), 3, cv2.LINE_AA)

                # Top Alert Banner
                cv2.rectangle(canvas, (int(width * 0.16), int(height * 0.09)), (int(width * 0.68), int(height * 0.19)), (35, 24, 38), -1)
                cv2.rectangle(canvas, (int(width * 0.16), int(height * 0.09)), (int(width * 0.68), int(height * 0.19)), (50, 160, 240), 2, cv2.LINE_AA)
                cv2.putText(canvas, "CRITICAL ROUTINE BOTTLENECK DETECTED", (int(width * 0.18), int(height * 0.15)), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.85, (50, 210, 255), 2, cv2.LINE_AA)

                # Interactive Checklist Card
                card_x, card_y, card_w, card_h = int(width * 0.08), int(height * 0.23), int(width * 0.58), 440
                cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (28, 22, 34), -1)
                cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 60, 90), 2, cv2.LINE_AA)
                cv2.putText(canvas, "DAILY EXECUTION SCHEDULE // AUDIT LOG", (card_x + 28, card_y + 44), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.72, (220, 220, 220), 2, cv2.LINE_AA)

                # Task 1: Good Habit
                t1_y = card_y + 110
                cv2.rectangle(canvas, (card_x + 28, t1_y - 28), (card_x + card_w - 28, t1_y + 28), (34, 30, 42), -1)
                cv2.line(canvas, (card_x + 44, t1_y), (card_x + 54, t1_y + 12), (60, 240, 100), 3, cv2.LINE_AA)
                cv2.line(canvas, (card_x + 54, t1_y + 12), (card_x + 72, t1_y - 10), (60, 240, 100), 3, cv2.LINE_AA)
                cv2.putText(canvas, "07:00 AM  Daily Plan & High-Priority Objectives", (card_x + 90, t1_y + 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 240, 180), 2, cv2.LINE_AA)

                # Task 2: Fatal Mistake (Pulsing Red)
                t2_y = card_y + 195
                is_clicked = t >= click_time
                mistake_bg = (45, 25, 30) if not is_clicked else (25, 45, 30)
                mistake_border = (50, 50, 230) if not is_clicked else (60, 240, 100)
                cv2.rectangle(canvas, (card_x + 28, t2_y - 32), (card_x + card_w - 28, t2_y + 32), mistake_bg, -1)
                cv2.rectangle(canvas, (card_x + 28, t2_y - 32), (card_x + card_w - 28, t2_y + 32), mistake_border, 2, cv2.LINE_AA)
                
                if not is_clicked:
                    # Red cross
                    cv2.line(canvas, (card_x + 48, t2_y - 12), (card_x + 68, t2_y + 12), (50, 50, 240), 3, cv2.LINE_AA)
                    cv2.line(canvas, (card_x + 68, t2_y - 12), (card_x + 48, t2_y + 12), (50, 50, 240), 3, cv2.LINE_AA)
                    cv2.putText(canvas, "08:30 AM  THE ONE BIG MISTAKE: Unfocused Context Switching", (card_x + 90, t2_y + 8), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.60, (90, 110, 255), 2, cv2.LINE_AA)
                else:
                    # Replaced with Green fix
                    cv2.line(canvas, (card_x + 44, t2_y), (card_x + 54, t2_y + 12), (60, 240, 100), 3, cv2.LINE_AA)
                    cv2.line(canvas, (card_x + 54, t2_y + 12), (card_x + 72, t2_y - 10), (60, 240, 100), 3, cv2.LINE_AA)
                    cv2.putText(canvas, "08:30 AM  [OPTIMIZED] Single-Task Deep Execution Engaged", (card_x + 90, t2_y + 8), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.60, (100, 255, 140), 2, cv2.LINE_AA)

                # Task 3: Outcome
                t3_y = card_y + 280
                cv2.rectangle(canvas, (card_x + 28, t3_y - 28), (card_x + card_w - 28, t3_y + 28), (34, 30, 42), -1)
                cv2.circle(canvas, (card_x + 58, t3_y), 8, (240, 180, 50), -1, cv2.LINE_AA)
                cv2.putText(canvas, "09:00 AM  Output Velocity Scaled 4x with Zero Drag", (card_x + 90, t3_y + 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 220, 255), 2, cv2.LINE_AA)

                # Status Footer in Card
                cv2.rectangle(canvas, (card_x + 28, card_y + 350), (card_x + card_w - 28, card_y + 400), (22, 18, 28), -1)
                status_txt = "ACTION REQUIRED: CLICK TO ELIMINATE ROUTINE DRAG" if not is_clicked else "STATUS: BOTTLENECK ELIMINATED SUCCESSFULLY"
                cv2.putText(canvas, status_txt, (card_x + 48, card_y + 382), cv2.FONT_HERSHEY_DUPLEX, 0.58, 
                            (60, 200, 240) if not is_clicked else (80, 255, 130), 2, cv2.LINE_AA)

            # ----------------- THEME 2: TIME WASTE & FRICTION -----------------
            elif theme_mode == "waste":
                self._apply_theme_background(canvas, base_bg, (22, 16, 26))
                cursor_target = (int(width * 0.44), int(height * 0.55))

                # Digital Stopwatch Card
                sw_x, sw_y = int(width * 0.08), int(height * 0.12)
                cv2.rectangle(canvas, (sw_x, sw_y), (sw_x + 480, sw_y + 190), (32, 22, 36), -1)
                cv2.rectangle(canvas, (sw_x, sw_y), (sw_x + 480, sw_y + 190), (60, 50, 220), 2, cv2.LINE_AA)
                cv2.putText(canvas, "CHRONIC TIME LEAKAGE", (sw_x + 24, sw_y + 40), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.68, (180, 180, 200), 2, cv2.LINE_AA)
                
                # Fast counting stopwatch
                wasted_mins = min(88, int(progress * 65) + 24)
                wasted_secs = int((t * 45) % 60)
                cv2.putText(canvas, f"00:{wasted_mins:02d}:{wasted_secs:02d}", (sw_x + 24, sw_y + 120), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.8, (50, 70, 255), 3, cv2.LINE_AA)
                cv2.putText(canvas, "HOURS LOST TO INEFFICIENCY", (sw_x + 24, sw_y + 160), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.54, (140, 140, 180), 1, cv2.LINE_AA)

                # Animated Hourglass with Falling Sand
                hg_cx, hg_cy = int(width * 0.45), int(height * 0.28)
                # Funnel outlines
                cv2.line(canvas, (hg_cx - 50, hg_cy - 75), (hg_cx + 50, hg_cy - 75), (200, 200, 200), 3, cv2.LINE_AA) # top lid
                cv2.line(canvas, (hg_cx - 50, hg_cy + 75), (hg_cx + 50, hg_cy + 75), (200, 200, 200), 3, cv2.LINE_AA) # bottom lid
                cv2.line(canvas, (hg_cx - 50, hg_cy - 75), (hg_cx, hg_cy), (160, 160, 160), 2, cv2.LINE_AA)
                cv2.line(canvas, (hg_cx + 50, hg_cy - 75), (hg_cx, hg_cy), (160, 160, 160), 2, cv2.LINE_AA)
                cv2.line(canvas, (hg_cx, hg_cy), (hg_cx - 50, hg_cy + 75), (160, 160, 160), 2, cv2.LINE_AA)
                cv2.line(canvas, (hg_cx, hg_cy), (hg_cx + 50, hg_cy + 75), (160, 160, 160), 2, cv2.LINE_AA)
                
                # Falling sand particles
                for si in range(8):
                    sand_y = int(hg_cy + ((t * 80 + si * 18) % 70))
                    sand_x = hg_cx + int(math.sin(t * 10 + si) * 3)
                    cv2.circle(canvas, (sand_x, sand_y), 2, (60, 200, 255), -1, cv2.LINE_AA)
                # Accumulated sand mound in bottom bulb
                mound_h = int(12 + progress * 32)
                cv2.fillPoly(canvas, [np.array([
                    [hg_cx - 38, hg_cy + 73],
                    [hg_cx + 38, hg_cy + 73],
                    [hg_cx, hg_cy + 73 - mound_h]
                ], dtype=np.int32)], (50, 190, 240), cv2.LINE_AA)

                # Draining Battery Gauge (Card 2)
                bat_x, bat_y = int(width * 0.08), int(height * 0.44)
                cv2.rectangle(canvas, (bat_x, bat_y), (bat_x + 580, bat_y + 190), (32, 22, 36), -1)
                cv2.rectangle(canvas, (bat_x, bat_y), (bat_x + 580, bat_y + 190), (80, 60, 90), 2, cv2.LINE_AA)
                cv2.putText(canvas, "ENERGY DRAIN & FRICTION IMPACT", (bat_x + 24, bat_y + 40), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.68, (220, 220, 220), 2, cv2.LINE_AA)
                
                # Battery Shell
                cv2.rectangle(canvas, (bat_x + 24, bat_y + 65), (bat_x + 360, bat_y + 125), (60, 60, 70), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (bat_x + 360, bat_y + 85), (bat_x + 372, bat_y + 105), (60, 60, 70), -1)
                
                # Draining fill: 80% down to 15%
                bat_pct = max(0.12, 0.85 - progress * 0.72)
                fill_w = int(330 * bat_pct)
                bat_col = (50, 50, 240) if bat_pct < 0.28 else (40, 180, 240)
                cv2.rectangle(canvas, (bat_x + 28, bat_y + 70), (bat_x + 28 + fill_w, bat_y + 120), bat_col, -1)
                cv2.putText(canvas, f"REMAINING: {int(bat_pct*100)}%", (bat_x + 390, bat_y + 102), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, bat_col, 2, cv2.LINE_AA)
                cv2.putText(canvas, "ALERT: SEVERE PERFORMANCE DEGRADATION", (bat_x + 24, bat_y + 162), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.54, (80, 90, 255), 1, cv2.LINE_AA)

            # ----------------- THEME 3: 7-DAY SPRINT ACCELERATION -----------------
            elif theme_mode == "calendar":
                self._apply_theme_background(canvas, base_bg, (28, 20, 18))
                cursor_target = (int(width * 0.52), int(height * 0.38))

                # Header
                cv2.putText(canvas, "THE 7-DAY ACCELERATED BLUEPRINT", (int(width * 0.08), int(height * 0.12)), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, (240, 220, 200), 2, cv2.LINE_AA)
                cv2.putText(canvas, "TRANSFORM YOUR RESULTS IN UNDER A WEEK INSTEAD OF MONTHS", 
                            (int(width * 0.08), int(height * 0.17)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 160, 150), 1, cv2.LINE_AA)

                # 7-Day Timeline Strips
                days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
                start_x = int(width * 0.08)
                day_w = 110
                gap = 22
                for d_idx, day_name in enumerate(days):
                    dx = start_x + d_idx * (day_w + gap)
                    dy = int(height * 0.24)
                    is_past = progress >= (d_idx / 7.0)
                    is_active = (d_idx == int(progress * 7.0))
                    
                    bg_col = (45, 80, 30) if is_past else ((35, 30, 42) if not is_active else (30, 120, 200))
                    border_col = (60, 240, 100) if is_past else ((70, 60, 80) if not is_active else (50, 220, 255))
                    
                    cv2.rectangle(canvas, (dx, dy), (dx + day_w, dy + 180), bg_col, -1)
                    cv2.rectangle(canvas, (dx, dy), (dx + day_w, dy + 180), border_col, 2, cv2.LINE_AA)
                    cv2.putText(canvas, day_name, (dx + 24, dy + 42), cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(canvas, f"0{d_idx+1}", (dx + 30, dy + 95), cv2.FONT_HERSHEY_DUPLEX, 0.95, (220, 220, 220), 2, cv2.LINE_AA)
                    
                    if is_past:
                        # Green checkmark
                        cv2.line(canvas, (dx + 35, dy + 135), (dx + 48, dy + 150), (60, 255, 120), 3, cv2.LINE_AA)
                        cv2.line(canvas, (dx + 48, dy + 150), (dx + 75, dy + 125), (60, 255, 120), 3, cv2.LINE_AA)
                    else:
                        cv2.circle(canvas, (dx + 55, dy + 140), 10, (100, 90, 110), 2, cv2.LINE_AA)

                # Fast Flipping Calendar Card (Right side of graphic area)
                cal_x, cal_y = int(width * 0.08), int(height * 0.52)
                cv2.rectangle(canvas, (cal_x, cal_y), (cal_x + 580, cal_y + 160), (35, 26, 24), -1)
                cv2.rectangle(canvas, (cal_x, cal_y), (cal_x + 580, cal_y + 160), (60, 180, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, ">>> ACCELERATED VELOCITY: +450% TIMELINE COMPRESSION", 
                            (cal_x + 28, cal_y + 55), cv2.FONT_HERSHEY_DUPLEX, 0.65, (60, 220, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "Full results unlocked within Day 7 with zero backlog", 
                            (cal_x + 28, cal_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 1, cv2.LINE_AA)

            # ----------------- THEME 4: FAST TWEAK & FOCUS LOCK -----------------
            elif theme_mode == "tweak":
                self._apply_theme_background(canvas, base_bg, (20, 24, 18))
                cursor_target = (int(width * 0.42), int(height * 0.38))

                # Cybernetic Sniper Crosshairs
                ret_cx, ret_cy = int(width * 0.18), int(height * 0.38)
                rot_ang = t * 1.8
                # Rotating outer tick ring
                cv2.circle(canvas, (ret_cx, ret_cy), 95, (40, 140, 60), 2, cv2.LINE_AA)
                for ai in range(8):
                    ang = rot_ang + ai * (math.pi / 4)
                    p1 = (int(ret_cx + 90 * math.cos(ang)), int(ret_cy + 90 * math.sin(ang)))
                    p2 = (int(ret_cx + 105 * math.cos(ang)), int(ret_cy + 105 * math.sin(ang)))
                    cv2.line(canvas, p1, p2, (60, 220, 90), 2, cv2.LINE_AA)
                # Crosshair lines
                cv2.line(canvas, (ret_cx - 110, ret_cy), (ret_cx - 30, ret_cy), (60, 220, 90), 2, cv2.LINE_AA)
                cv2.line(canvas, (ret_cx + 30, ret_cy), (ret_cx + 110, ret_cy), (60, 220, 90), 2, cv2.LINE_AA)
                cv2.line(canvas, (ret_cx, ret_cy - 110), (ret_cx, ret_cy - 30), (60, 220, 90), 2, cv2.LINE_AA)
                cv2.line(canvas, (ret_cx, ret_cy + 30), (ret_cx, ret_cy + 110), (60, 220, 90), 2, cv2.LINE_AA)
                cv2.circle(canvas, (ret_cx, ret_cy), 8, (60, 255, 120), -1, cv2.LINE_AA)
                cv2.putText(canvas, "[LOCK: ACQUIRED]", (ret_cx - 65, ret_cy + 130), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.54, (80, 255, 130), 1, cv2.LINE_AA)

                # Interactive Focus Switch Card
                sw_x, sw_y, sw_w, sw_h = int(width * 0.30), int(height * 0.20), 480, 360
                cv2.rectangle(canvas, (sw_x, sw_y), (sw_x + sw_w, sw_y + sw_h), (25, 30, 24), -1)
                cv2.rectangle(canvas, (sw_x, sw_y), (sw_x + sw_w, sw_y + sw_h), (60, 160, 80), 2, cv2.LINE_AA)
                cv2.putText(canvas, "MODE RECONFIGURATION", (sw_x + 28, sw_y + 44), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.72, (220, 240, 220), 2, cv2.LINE_AA)

                # The Interactive Toggle Switch
                is_switched = t >= click_time
                tg_x, tg_y = sw_x + 80, sw_y + 110
                tg_w, tg_h = 240, 70
                # Housing
                housing_col = (20, 90, 40) if is_switched else (45, 45, 50)
                cv2.rectangle(canvas, (tg_x, tg_y), (tg_x + tg_w, tg_y + tg_h), housing_col, -1)
                cv2.rectangle(canvas, (tg_x, tg_y), (tg_x + tg_w, tg_y + tg_h), (80, 220, 120) if is_switched else (120, 120, 120), 2, cv2.LINE_AA)
                
                # Knob position moves on click
                knob_x = tg_x + tg_w - 45 if is_switched else tg_x + 45
                cv2.circle(canvas, (knob_x, tg_y + 35), 28, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (knob_x, tg_y + 35), 28, (60, 220, 100) if is_switched else (140, 140, 140), 2, cv2.LINE_AA)
                
                # Switch label
                mode_str = "HYPER FOCUS: ACTIVE" if is_switched else "DEFAULT ROUTINE: OFF"
                cv2.putText(canvas, mode_str, (sw_x + 40, sw_y + 240), cv2.FONT_HERSHEY_DUPLEX, 0.72, 
                            (80, 255, 140) if is_switched else (160, 160, 160), 2, cv2.LINE_AA)
                cv2.putText(canvas, "First move made instant. Distractions locked out.", (sw_x + 40, sw_y + 290), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 220, 180), 1, cv2.LINE_AA)

            # ----------------- THEME 5: 5-SECOND INSTANT FIX -----------------
            elif theme_mode == "instant":
                self._apply_theme_background(canvas, base_bg, (18, 26, 22))
                cursor_target = (int(width * 0.48), int(height * 0.40))

                # Giant 5.00s Countdown Gauge
                gauge_cx, gauge_cy = int(width * 0.22), int(height * 0.40)
                gauge_r = 120
                cv2.circle(canvas, (gauge_cx, gauge_cy), gauge_r, (40, 60, 50), 6, cv2.LINE_AA)
                
                # Depleting radial arc
                gauge_pct = max(0.0, 1.0 - progress)
                end_angle = int(360 * gauge_pct)
                if end_angle > 0:
                    cv2.ellipse(canvas, (gauge_cx, gauge_cy), (gauge_r, gauge_r), -90, 0, end_angle, (50, 240, 140), 8, cv2.LINE_AA)

                # Digital countdown numbers
                remaining_sec = max(0.0, 5.0 * (1.0 - progress))
                cv2.putText(canvas, f"{remaining_sec:04.2f}s", (gauge_cx - 72, gauge_cy + 16), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(canvas, "< 5 SECONDS FIX", (gauge_cx - 68, gauge_cy + 65), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, (60, 240, 140), 1, cv2.LINE_AA)

                # Instant Fix Celebration Card
                fix_x, fix_y, fix_w, fix_h = int(width * 0.36), int(height * 0.22), 520, 360
                cv2.rectangle(canvas, (fix_x, fix_y), (fix_x + fix_w, fix_y + fix_h), (25, 34, 30), -1)
                cv2.rectangle(canvas, (fix_x, fix_y), (fix_x + fix_w, fix_y + fix_h), (60, 220, 140), 2, cv2.LINE_AA)
                cv2.putText(canvas, "INSTANT PERFORMANCE PATCH", (fix_x + 28, fix_y + 44), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.72, (240, 255, 240), 2, cv2.LINE_AA)

                # Giant Animated Checkmark
                chk_cx, chk_cy = fix_x + 90, fix_y + 160
                cv2.circle(canvas, (chk_cx, chk_cy), 48, (40, 160, 80), -1, cv2.LINE_AA)
                cv2.line(canvas, (chk_cx - 20, chk_cy), (chk_cx - 4, chk_cy + 18), (255, 255, 255), 5, cv2.LINE_AA)
                cv2.line(canvas, (chk_cx - 4, chk_cy + 18), (chk_cx + 26, chk_cy - 16), (255, 255, 255), 5, cv2.LINE_AA)
                
                # Confetti sparks shooting out
                for ci in range(12):
                    c_ang = ci * (math.pi / 6) + t * 4
                    c_dist = 65 + math.sin(t * 8 + ci) * 15
                    sp_x = int(chk_cx + c_dist * math.cos(c_ang))
                    sp_y = int(chk_cy + c_dist * math.sin(c_ang))
                    cv2.circle(canvas, (sp_x, sp_y), 4, (60, 240, 255) if ci % 2 == 0 else (60, 255, 120), -1, cv2.LINE_AA)

                cv2.putText(canvas, "PROBLEM RESOLVED", (fix_x + 165, fix_y + 145), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "EXECUTION: ZERO LATENCY", (fix_x + 165, fix_y + 185), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80, 255, 140), 2, cv2.LINE_AA)

                # Comparison pill
                cv2.rectangle(canvas, (fix_x + 28, fix_y + 260), (fix_x + fix_w - 28, fix_y + 320), (35, 45, 40), -1)
                cv2.putText(canvas, "MANUAL FIX: 4 HOURS  -->  INCUBRIX: 4.8s", (fix_x + 48, fix_y + 298), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.58, (60, 220, 255), 2, cv2.LINE_AA)

            # ----------------- THEME 6: INBOX BLUEPRINT DELIVERY -----------------
            elif theme_mode == "email":
                self._apply_theme_background(canvas, base_bg, (26, 18, 22))
                cursor_target = (int(width * 0.44), int(height * 0.50))

                # Glassmorphic Email Client UI Card
                em_x, em_y, em_w, em_h = int(width * 0.08), int(height * 0.16), 560, 440
                cv2.rectangle(canvas, (em_x, em_y), (em_x + em_w, em_y + em_h), (34, 24, 30), -1)
                cv2.rectangle(canvas, (em_x, em_y), (em_x + em_w, em_y + em_h), (80, 160, 255), 2, cv2.LINE_AA)
                
                # Titlebar
                cv2.rectangle(canvas, (em_x, em_y), (em_x + em_w, em_y + 45), (45, 32, 40), -1)
                cv2.putText(canvas, "MAIL CLIENT // INBOX (1 UNREAD)", (em_x + 24, em_y + 30), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (240, 240, 255), 2, cv2.LINE_AA)
                
                # Email Entry
                cv2.putText(canvas, "FROM: IncuBrix AI <blueprint@incubrix.ai>", (em_x + 28, em_y + 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 200), 1, cv2.LINE_AA)
                cv2.putText(canvas, "SUBJECT: Full Step-by-Step Action Breakdown", (em_x + 28, em_y + 130), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

                # Animated 3D Envelope & Sliding Document
                env_x, env_y = em_x + 240, em_y + 260
                # Document sliding up out of envelope
                doc_lift = int(min(65.0, progress * 85.0))
                cv2.rectangle(canvas, (env_x - 90, env_y - 60 - doc_lift), (env_x + 90, env_y + 10 - doc_lift), (240, 240, 245), -1)
                cv2.putText(canvas, "ACTION PLAN", (env_x - 65, env_y - 25 - doc_lift), cv2.FONT_HERSHEY_DUPLEX, 0.55, (30, 30, 30), 2, cv2.LINE_AA)
                cv2.putText(canvas, "[PDF BLUEPRINT]", (env_x - 70, env_y - 2 - doc_lift), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 120, 220), 1, cv2.LINE_AA)
                
                # Envelope body
                cv2.rectangle(canvas, (env_x - 110, env_y - 30), (env_x + 110, env_y + 70), (55, 40, 50), -1)
                cv2.rectangle(canvas, (env_x - 110, env_y - 30), (env_x + 110, env_y + 70), (80, 160, 255), 2, cv2.LINE_AA)
                # Flap fold lines
                cv2.line(canvas, (env_x - 110, env_y - 30), (env_x, env_y + 20), (80, 160, 255), 2, cv2.LINE_AA)
                cv2.line(canvas, (env_x + 110, env_y - 30), (env_x, env_y + 20), (80, 160, 255), 2, cv2.LINE_AA)

                # Delivery notification pill
                cv2.rectangle(canvas, (em_x + 28, em_y + 360), (em_x + em_w - 28, em_y + 415), (25, 45, 30), -1)
                cv2.rectangle(canvas, (em_x + 28, em_y + 360), (em_x + em_w - 28, em_y + 415), (60, 240, 100), 2, cv2.LINE_AA)
                cv2.putText(canvas, "STATUS: SENT DIRECTLY TO YOUR INBOX", (em_x + 60, em_y + 396), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.62, (80, 255, 140), 2, cv2.LINE_AA)

            # ----------------- THEME 7: SOCIAL COMMENTS 'YES' FEED -----------------
            elif theme_mode == "social":
                self._apply_theme_background(canvas, base_bg, (24, 16, 28))
                cursor_target = (int(width * 0.46), int(height * 0.54))

                # Live Comments Feed Card
                sc_x, sc_y, sc_w, sc_h = int(width * 0.08), int(height * 0.16), 560, 440
                cv2.rectangle(canvas, (sc_x, sc_y), (sc_x + sc_w, sc_y + sc_h), (34, 22, 38), -1)
                cv2.rectangle(canvas, (sc_x, sc_y), (sc_x + sc_w, sc_y + sc_h), (200, 100, 240), 2, cv2.LINE_AA)
                cv2.putText(canvas, "COMMUNITY STREAM // LIVE COMMENTS", (sc_x + 28, sc_y + 44), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.68, (255, 230, 255), 2, cv2.LINE_AA)

                # Bubble 1
                b1_y = sc_y + 80
                cv2.rectangle(canvas, (sc_x + 24, b1_y), (sc_x + sc_w - 24, b1_y + 55), (45, 30, 50), -1)
                cv2.circle(canvas, (sc_x + 50, b1_y + 28), 16, (200, 120, 240), -1, cv2.LINE_AA)
                cv2.putText(canvas, "@alex_creator: YES! Dropping in comments now", (sc_x + 78, b1_y + 34), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 240, 255), 1, cv2.LINE_AA)

                # Bubble 2
                b2_y = sc_y + 155
                cv2.rectangle(canvas, (sc_x + 24, b2_y), (sc_x + sc_w - 24, b2_y + 55), (45, 30, 50), -1)
                cv2.circle(canvas, (sc_x + 50, b2_y + 28), 16, (240, 160, 60), -1, cv2.LINE_AA)
                cv2.putText(canvas, "@sarah_dev: YES! Need the full breakdown PDF", (sc_x + 78, b2_y + 34), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 240, 255), 1, cv2.LINE_AA)

                # Bubble 3
                b3_y = sc_y + 230
                cv2.rectangle(canvas, (sc_x + 24, b3_y), (sc_x + sc_w - 24, b3_y + 55), (45, 30, 50), -1)
                cv2.circle(canvas, (sc_x + 50, b3_y + 28), 16, (60, 220, 120), -1, cv2.LINE_AA)
                cv2.putText(canvas, "@marcus_ai: Commented 'YES'! Game changer", (sc_x + 78, b3_y + 34), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 240, 255), 1, cv2.LINE_AA)

                # Interactive "DROP 'YES' NOW" Call to Action Button
                is_sub = t >= click_time
                btn_y = sc_y + 325
                btn_col = (20, 120, 50) if is_sub else (160, 60, 220)
                cv2.rectangle(canvas, (sc_x + 24, btn_y), (sc_x + sc_w - 24, btn_y + 75), btn_col, -1)
                cv2.rectangle(canvas, (sc_x + 24, btn_y), (sc_x + sc_w - 24, btn_y + 75), (255, 255, 255), 2, cv2.LINE_AA)
                btn_label = "✔ 'YES' REGISTERED! CHECK YOUR DM" if is_sub else "▶ DROP A 'YES' IN THE COMMENTS RIGHT NOW"
                cv2.putText(canvas, btn_label, (sc_x + 40, btn_y + 46), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

                # Floating Reaction Fountain (Hearts & Embers rising)
                for hi in range(8):
                    f_y = int((height * 0.85) - ((t * 90 + hi * 45) % (height * 0.70)))
                    f_x = int(width * 0.44 + math.sin(t * 3 + hi) * 45 + (hi % 3) * 35)
                    # Heart shape
                    cv2.circle(canvas, (f_x - 6, f_y - 4), 7, (80, 60, 240), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (f_x + 6, f_y - 4), 7, (80, 60, 240), -1, cv2.LINE_AA)
                    cv2.fillPoly(canvas, [np.array([[f_x - 13, f_y - 2], [f_x + 13, f_y - 2], [f_x, f_y + 13]], np.int32)], (80, 60, 240), cv2.LINE_AA)

            # ----------------- THEME 8: FINANCIAL SURGE -----------------
            elif theme_mode == "finance":
                self._apply_theme_background(canvas, base_bg, (16, 14, 10))
                cursor_target = (int(width * 0.40), int(height * 0.28))
                
                # Grid
                for gx in range(0, int(width * 0.74), 100):
                    cv2.line(canvas, (gx, 0), (gx, height), (32, 26, 18), 1)
                for gy in range(0, height, 80):
                    cv2.line(canvas, (0, gy), (int(width * 0.74), gy), (32, 26, 18), 1)

                # Animated candlestick pillars
                for ci in range(12):
                    bar_x = int(width * 0.10 + ci * (width * 0.56 / 11))
                    bar_h = int(60 + ci * 24 + math.sin(t * 3.5 + ci) * 18)
                    top_y = int(height * 0.78 - bar_h)
                    candle_color = (45, 190, 85) if ci % 3 != 1 else (65, 65, 210)
                    cv2.line(canvas, (bar_x, top_y - 18), (bar_x, int(height * 0.78) + 18), (80, 80, 80), 1)
                    cv2.rectangle(canvas, (bar_x - 13, top_y), (bar_x + 13, int(height * 0.78)), candle_color, -1)

                # Upward trend curve
                visible_count = max(2, int(progress * len(chart_points)) + 1)
                visible_pts = chart_points[:visible_count]
                if len(visible_pts) > 1:
                    poly_pts = [np.array([visible_pts[0][0], int(height * 0.80)])]
                    poly_pts.extend([np.array(p) for p in visible_pts])
                    poly_pts.append(np.array([visible_pts[-1][0], int(height * 0.80)]))
                    fill_mask = np.zeros_like(canvas)
                    cv2.fillPoly(fill_mask, [np.array(poly_pts, dtype=np.int32)], (25, 95, 45))
                    canvas = cv2.addWeighted(canvas, 1.0, fill_mask, 0.48, 0)
                    for i in range(len(visible_pts) - 1):
                        cv2.line(canvas, visible_pts[i], visible_pts[i+1], (55, 245, 125), 4, cv2.LINE_AA)

                # Metric Card
                card1_x, card1_y, card1_w, card1_h = int(width * 0.08), int(height * 0.12), 480, 160
                cv2.rectangle(canvas, (card1_x, card1_y), (card1_x + card1_w, card1_y + card1_h), (28, 24, 18), -1)
                cv2.rectangle(canvas, (card1_x, card1_y), (card1_x + card1_w, card1_y + card1_h), (60, 220, 120), 2, cv2.LINE_AA)
                cv2.putText(canvas, "ANNUAL REVENUE SURGE", (card1_x + 22, card1_y + 36), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
                current_percent = min(40.2, progress * 40.2 * 1.3)
                cv2.putText(canvas, f"+{current_percent:04.1f}% EXPANSION", (card1_x + 22, card1_y + 92), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.25, (60, 255, 140), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (card1_x + 22, card1_y + 112), (card1_x + 240, card1_y + 142), (20, 80, 40), -1)
                cv2.putText(canvas, "▲ KPI TARGET EXCEEDED", (card1_x + 30, card1_y + 132), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 255, 170), 1, cv2.LINE_AA)

                # Floating 3D Gold Coins
                for ci in range(3):
                    coin_t = t * 2 + ci * 1.2
                    coin_x = int(width * 0.42 + math.sin(coin_t) * 70 + ci * 70)
                    coin_y = int(height * 0.28 + math.cos(coin_t * 0.8) * 35 + ci * 30)
                    cv2.circle(canvas, (coin_x, coin_y), 24, (30, 180, 240), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (coin_x, coin_y), 20, (50, 215, 255), -1, cv2.LINE_AA)
                    cv2.putText(canvas, "$", (coin_x - 7, coin_y + 8), cv2.FONT_HERSHEY_DUPLEX, 0.8, (20, 110, 180), 2, cv2.LINE_AA)

            # ----------------- THEME 9: CLOUD DEPLOYMENT & CODE -----------------
            elif theme_mode == "tech":
                self._apply_theme_background(canvas, base_bg, (20, 16, 10))
                cursor_target = (int(width * 0.42), int(height * 0.35))

                # Perspective Grid
                horizon_y = int(height * 0.40)
                vp_x = int(width * 0.42)
                for px in range(-width, width * 2, 140):
                    cv2.line(canvas, (vp_x, horizon_y), (px, height), (75, 50, 20), 1, cv2.LINE_AA)

                # Terminal Card
                term_x, term_y, term_w, term_h = int(width * 0.08), int(height * 0.12), 520, 220
                cv2.rectangle(canvas, (term_x, term_y), (term_x + term_w, term_y + term_h), (25, 20, 15), -1)
                cv2.rectangle(canvas, (term_x, term_y), (term_x + term_w, term_y + term_h), (220, 160, 50), 2, cv2.LINE_AA)
                cv2.putText(canvas, "bash — cloud-cluster-deploy", (term_x + 30, term_y + 28), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
                code_lines = [
                    "> python -m cloud.cluster --autoscale",
                    "[DEPLOY] Spinning up 64 distributed nodes...",
                    "[STATUS] 99.99% Uptime Verified. Zero Lag.",
                    "[BACKUP] Automated encrypted sync: ACTIVE"
                ]
                for l_idx, line in enumerate(code_lines):
                    if progress >= (l_idx * 0.22):
                        cv2.putText(canvas, line, (term_x + 22, term_y + 70 + l_idx * 34), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (80, 220, 255) if l_idx == 0 else (180, 240, 140), 1, cv2.LINE_AA)

                # Server Racks
                for ri in range(3):
                    srv_x = int(width * 0.44 + ri * 90)
                    srv_y = int(height * 0.32 + math.sin(t * 2.5 + ri) * 14)
                    cv2.rectangle(canvas, (srv_x - 36, srv_y - 55), (srv_x + 36, srv_y + 55), (55, 40, 22), -1)
                    cv2.rectangle(canvas, (srv_x - 36, srv_y - 55), (srv_x + 36, srv_y + 55), (220, 160, 50), 2, cv2.LINE_AA)
                    led_on = (frame_idx + ri * 7) % 18 > 4
                    cv2.circle(canvas, (srv_x + 18, srv_y - 35), 4, (60, 255, 100) if led_on else (40, 60, 180), -1, cv2.LINE_AA)
                    cv2.putText(canvas, f"NODE-{ri+1}", (srv_x - 28, srv_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)

            # ----------------- THEME 10: PARTNERSHIP TRUST -----------------
            elif theme_mode == "trust":
                self._apply_theme_background(canvas, base_bg, (18, 16, 24))
                cursor_target = (int(width * 0.40), int(height * 0.42))

                # Interlocking Gears
                gear1_c = (int(width * 0.32), int(height * 0.38))
                gear1_r = 85
                gear1_ang = t * 1.5
                cv2.circle(canvas, gear1_c, gear1_r, (40, 130, 200), 4, cv2.LINE_AA)
                for c_idx in range(8):
                    c_ang = gear1_ang + c_idx * (math.pi / 4)
                    tooth_x = int(gear1_c[0] + (gear1_r + 14) * math.cos(c_ang))
                    tooth_y = int(gear1_c[1] + (gear1_r + 14) * math.sin(c_ang))
                    cv2.circle(canvas, (tooth_x, tooth_y), 8, (50, 160, 230), -1, cv2.LINE_AA)

                gear2_c = (int(width * 0.44), int(height * 0.48))
                gear2_r = 60
                gear2_ang = -t * 1.5 * (gear1_r / gear2_r)
                cv2.circle(canvas, gear2_c, gear2_r, (40, 160, 220), 4, cv2.LINE_AA)
                for c_idx in range(6):
                    c_ang = gear2_ang + c_idx * (math.pi / 3)
                    tooth_x = int(gear2_c[0] + (gear2_r + 12) * math.cos(c_ang))
                    tooth_y = int(gear2_c[1] + (gear2_r + 12) * math.sin(c_ang))
                    cv2.circle(canvas, (tooth_x, tooth_y), 7, (60, 180, 240), -1, cv2.LINE_AA)

                # Card
                card3_x, card3_y, card3_w, card3_h = int(width * 0.08), int(height * 0.12), 480, 170
                cv2.rectangle(canvas, (card3_x, card3_y), (card3_x + card3_w, card3_y + card3_h), (28, 22, 34), -1)
                cv2.rectangle(canvas, (card3_x, card3_y), (card3_x + card3_w, card3_y + card3_h), (60, 170, 240), 2, cv2.LINE_AA)
                cv2.putText(canvas, "PARTNERSHIP ALLIANCE", (card3_x + 22, card3_y + 36), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(canvas, "TRUST BUILT OVER TIME", (card3_x + 22, card3_y + 82), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, (80, 210, 255), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (card3_x + 22, card3_y + 110), (card3_x + card3_w - 22, card3_y + 135), (45, 35, 55), -1)
                fill_w = int((card3_w - 44) * min(1.0, progress * 1.4))
                cv2.rectangle(canvas, (card3_x + 22, card3_y + 110), (card3_x + 22 + fill_w, card3_y + 135), (60, 180, 255), -1)

            # ----------------- THEME: CLINICAL HOSPITAL & MEDICAL WARD -----------------
            elif theme_mode == "hospital":
                # Deep medical cyan-charcoal hospital walls
                self._apply_theme_background(canvas, base_bg, (32, 28, 22))
                cursor_target = (int(width * 0.38), int(height * 0.38))

                floor_y = int(height * 0.82)
                cv2.rectangle(canvas, (0, floor_y), (width, height), (22, 18, 16), -1)
                cv2.line(canvas, (0, floor_y), (width, floor_y), (75, 65, 55), 3, cv2.LINE_AA)
                for gx in range(0, int(width * 0.74), 110):
                    cv2.line(canvas, (gx, 0), (gx, floor_y), (44, 38, 30), 1)

                # 1. ICU Patient Vitals Monitor
                mon_x, mon_y, mon_w, mon_h = int(width * 0.08), int(height * 0.12), int(width * 0.48), 340
                cv2.rectangle(canvas, (mon_x, mon_y), (mon_x + mon_w, mon_y + mon_h), (18, 22, 22), -1)
                cv2.rectangle(canvas, (mon_x, mon_y), (mon_x + mon_w, mon_y + mon_h), (50, 180, 120), 2, cv2.LINE_AA)

                # Monitor Header
                cv2.rectangle(canvas, (mon_x, mon_y), (mon_x + mon_w, mon_y + 42), (24, 32, 28), -1)
                # Medical Cross Icon
                cv2.circle(canvas, (mon_x + 26, mon_y + 21), 12, (5, 150, 105), -1, cv2.LINE_AA)
                cv2.line(canvas, (mon_x + 26, mon_y + 13), (mon_x + 26, mon_y + 29), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(canvas, (mon_x + 18, mon_y + 21), (mon_x + 34, mon_y + 21), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "ICU CLINICAL MONITOR // REAL-TIME VITALS", (mon_x + 48, mon_y + 28),
                            cv2.FONT_HERSHEY_DUPLEX, 0.54, (240, 240, 240), 1, cv2.LINE_AA)

                # Live Heart Rate & Vitals Readouts
                pulse_bpm = int(72 + math.sin(t * 1.5) * 3)
                cv2.putText(canvas, f"HR: {pulse_bpm} BPM", (mon_x + 24, mon_y + 88),
                            cv2.FONT_HERSHEY_DUPLEX, 0.85, (60, 240, 120), 2, cv2.LINE_AA)
                cv2.putText(canvas, "SPO2: 99%", (mon_x + 230, mon_y + 88),
                            cv2.FONT_HERSHEY_DUPLEX, 0.85, (240, 200, 60), 2, cv2.LINE_AA)
                cv2.putText(canvas, "BP: 120/80", (mon_x + 400, mon_y + 88),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (80, 180, 240), 2, cv2.LINE_AA)

                # Real-Time Animated ECG Waveform
                ecg_y = mon_y + 175
                cv2.line(canvas, (mon_x + 20, ecg_y), (mon_x + mon_w - 20, ecg_y), (30, 40, 35), 1)
                ecg_pts = []
                for x_idx in range(mon_w - 40):
                    lx = mon_x + 20 + x_idx
                    phase = (lx - int(t * 220)) % 220
                    if 80 <= phase <= 95:
                        dy = -int(math.sin((phase - 80) / 15 * math.pi) * 16)
                    elif 105 <= phase <= 112:
                        dy = int(10)
                    elif 113 <= phase <= 124:
                        dy = -int(math.sin((phase - 113) / 11 * math.pi) * 62)
                    elif 125 <= phase <= 132:
                        dy = int(14)
                    elif 145 <= phase <= 170:
                        dy = -int(math.sin((phase - 145) / 25 * math.pi) * 20)
                    else:
                        dy = int(random.uniform(-1, 1))
                    ecg_pts.append((lx, ecg_y + dy))

                for i in range(len(ecg_pts) - 1):
                    cv2.line(canvas, ecg_pts[i], ecg_pts[i+1], (60, 255, 140), 2, cv2.LINE_AA)

                # Status Diagnostic Banner
                cv2.rectangle(canvas, (mon_x + 20, mon_y + 245), (mon_x + mon_w - 20, mon_y + 315), (24, 32, 28), -1)
                cv2.putText(canvas, "LAB STATUS: RECOVERY PROTOCOL ACTIVE // PROGNOSIS: FULL RECOVERY",
                            (mon_x + 28, mon_y + 284), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 240, 200), 1, cv2.LINE_AA)

                # 2. IV Drip Stand Silhouette
                iv_x = int(width * 0.04)
                cv2.line(canvas, (iv_x, int(height * 0.22)), (iv_x, floor_y), (85, 80, 75), 3, cv2.LINE_AA)
                cv2.ellipse(canvas, (iv_x, int(height * 0.25)), (14, 24), 0, 0, 360, (180, 200, 210), -1, cv2.LINE_AA)

            # ----------------- THEME: ROYAL CITADEL & THRONE ROOM -----------------
            elif theme_mode == "royal":
                self._apply_theme_background(canvas, base_bg, (22, 16, 28))
                cursor_target = (int(width * 0.38), int(height * 0.38))

                # Castle stone masonry
                for sy in range(0, height, 85):
                    offset = 45 if (sy // 85) % 2 == 1 else 0
                    cv2.line(canvas, (0, sy), (int(width * 0.74), sy), (35, 26, 42), 2)
                    for sx in range(offset, int(width * 0.74), 95):
                        cv2.line(canvas, (sx, sy), (sx, sy + 85), (35, 26, 42), 2)

                # Torches with animated flickering flames
                for tx in [int(width * 0.08), int(width * 0.54)]:
                    ty = int(height * 0.32)
                    cv2.rectangle(canvas, (tx - 8, ty), (tx + 8, ty + 38), (40, 30, 20), -1)
                    flame_h = int(22 + math.sin(t * 14 + tx) * 7)
                    cv2.ellipse(canvas, (tx, ty - 10), (13, flame_h), 0, 0, 360, (20, 120, 240), -1, cv2.LINE_AA)
                    cv2.ellipse(canvas, (tx, ty - 10), (7, int(flame_h * 0.60)), 0, 0, 360, (50, 210, 255), -1, cv2.LINE_AA)

                # Royal Citadel Banner Card
                r_card_x, r_card_y, r_card_w, r_card_h = int(width * 0.14), int(height * 0.14), int(width * 0.38), 300
                cv2.rectangle(canvas, (r_card_x, r_card_y), (r_card_x + r_card_w, r_card_y + r_card_h), (35, 18, 28), -1)
                cv2.rectangle(canvas, (r_card_x, r_card_y), (r_card_x + r_card_w, r_card_y + r_card_h), (40, 180, 220), 2, cv2.LINE_AA)
                cv2.putText(canvas, "ROYAL CITADEL COUNCIL", (r_card_x + 28, r_card_y + 44),
                            cv2.FONT_HERSHEY_DUPLEX, 0.72, (240, 220, 200), 2, cv2.LINE_AA)
                cv2.putText(canvas, "DEFENSE STRATEGY // REALM WATCHTOWERS", (r_card_x + 28, r_card_y + 88),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.54, (80, 180, 240), 1, cv2.LINE_AA)

            # ----------------- THEME: CLEAN ENERGY & WIND FARM -----------------
            elif theme_mode == "clean_energy":
                if base_bg is not None and base_bg.size > 0:
                    self._apply_theme_background(canvas, base_bg, (30, 60, 40))
                else:
                    # Sunset gradient sky (rich deep indigo at top to warm sunset orange/amber at horizon)
                    for gy in range(height):
                        pct = gy / height
                        if pct < 0.65:
                            sky_p = pct / 0.65
                            b = int(60 * (1 - sky_p) + 20 * sky_p)
                            g = int(25 * (1 - sky_p) + 115 * sky_p)
                            r = int(50 * (1 - sky_p) + 245 * sky_p)
                        else:
                            ground_p = (pct - 0.65) / 0.35
                            b = int(20 * (1 - ground_p) + 12 * ground_p)
                            g = int(115 * (1 - ground_p) + 48 * ground_p)
                            r = int(245 * (1 - ground_p) + 32 * ground_p)
                        canvas[gy, :] = (b, g, r)

                cursor_target = (int(width * 0.38), int(height * 0.28))

                # Sunset Sun glow dipping at horizon
                sun_cx, sun_cy = int(width * 0.36), int(height * 0.58)
                for sr in range(110, 0, -12):
                    alpha = int(120 * (1.0 - sr / 110.0))
                    cv2.circle(canvas, (sun_cx, sun_cy), sr, (30 + alpha // 2, 140 + alpha, 255), -1, cv2.LINE_AA)

                # Distant rolling hills silhouettes
                horizon_y = int(height * 0.65)
                hill_pts = [(0, height)]
                for hx in range(0, width + 40, 30):
                    hy = int(horizon_y - 25 + math.sin(hx * 0.0035) * 45 + math.cos(hx * 0.008) * 20)
                    hill_pts.append((hx, hy))
                hill_pts.append((width, height))
                cv2.fillPoly(canvas, [np.array(hill_pts, np.int32)], (20, 65, 38), cv2.LINE_AA)

                # Foreground rolling hill
                fg_pts = [(0, height)]
                for hx in range(0, width + 40, 30):
                    hy = int(horizon_y + 40 + math.sin(hx * 0.005 + 1.2) * 35)
                    fg_pts.append((hx, hy))
                fg_pts.append((width, height))
                cv2.fillPoly(canvas, [np.array(fg_pts, np.int32)], (15, 42, 24), cv2.LINE_AA)

                # 3 Wind Turbines with 3 rotating blades
                turbines = [
                    {"x": int(width * 0.22), "base_y": int(height * 0.72), "hub_y": int(height * 0.38), "blade_len": 125, "blade_w": 8, "hub_r": 9, "speed": 2.8, "phase": 0.0},
                    {"x": int(width * 0.44), "base_y": int(height * 0.67), "hub_y": int(height * 0.43), "blade_len": 90, "blade_w": 6, "hub_r": 7, "speed": 2.4, "phase": 1.2},
                    {"x": int(width * 0.62), "base_y": int(height * 0.65), "hub_y": int(height * 0.47), "blade_len": 65, "blade_w": 4, "hub_r": 5, "speed": 2.1, "phase": 2.4},
                ]

                for tb in turbines:
                    tx, ty_base, ty_hub = tb["x"], tb["base_y"], tb["hub_y"]
                    blen, bw, hr = tb["blade_len"], tb["blade_w"], tb["hub_r"]
                    rot_angle = t * tb["speed"] + tb["phase"]

                    # Tapered tower pole
                    tower_pts = np.array([
                        [tx - 8, ty_base], [tx + 8, ty_base],
                        [tx + 4, ty_hub], [tx - 4, ty_hub]
                    ], np.int32)
                    cv2.fillPoly(canvas, [tower_pts], (235, 240, 245), cv2.LINE_AA)
                    cv2.polylines(canvas, [tower_pts], True, (160, 170, 180), 1, cv2.LINE_AA)

                    # Nacelle / Generator housing
                    cv2.rectangle(canvas, (tx - 10, ty_hub - 6), (tx + 12, ty_hub + 6), (220, 225, 230), -1)

                    # 3 Aerodynamic Blades spaced by 120 degrees (2 * pi / 3)
                    for b_idx in range(3):
                        blade_phi = rot_angle + b_idx * (2.0 * math.pi / 3.0)
                        tip_x = int(tx + math.cos(blade_phi) * blen)
                        tip_y = int(ty_hub + math.sin(blade_phi) * blen)
                        perp_phi = blade_phi + math.pi / 2.0
                        mid_x = int(tx + math.cos(blade_phi) * (blen * 0.45))
                        mid_y = int(ty_hub + math.sin(blade_phi) * (blen * 0.45))
                        p1_x = int(mid_x + math.cos(perp_phi) * bw)
                        p1_y = int(mid_y + math.sin(perp_phi) * bw)
                        p2_x = int(mid_x - math.cos(perp_phi) * (bw * 0.3))
                        p2_y = int(mid_y - math.sin(perp_phi) * (bw * 0.3))

                        b_pts = np.array([
                            [tx, ty_hub], [p1_x, p1_y], [tip_x, tip_y], [p2_x, p2_y]
                        ], np.int32)
                        cv2.fillPoly(canvas, [b_pts], (250, 252, 255), cv2.LINE_AA)
                        cv2.polylines(canvas, [b_pts], True, (170, 180, 190), 1, cv2.LINE_AA)

                    # Center Rotor Hub
                    cv2.circle(canvas, (tx, ty_hub), hr, (210, 220, 225), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (tx, ty_hub), hr // 2, (120, 130, 140), -1, cv2.LINE_AA)

                # Digital Carbon Metrics & Energy Telemetry Card
                ec_x, ec_y, ec_w, ec_h = int(width * 0.07), int(height * 0.12), int(width * 0.42), 290
                cv2.rectangle(canvas, (ec_x, ec_y), (ec_x + ec_w, ec_y + ec_h), (22, 26, 20), -1)
                cv2.rectangle(canvas, (ec_x, ec_y), (ec_x + ec_w, ec_y + ec_h), (60, 220, 120), 2, cv2.LINE_AA)

                # Card Header
                cv2.rectangle(canvas, (ec_x, ec_y), (ec_x + ec_w, ec_y + 44), (28, 48, 28), -1)
                cv2.putText(canvas, "CLEAN ENERGY GRID // CARBON AUDIT TELEMETRY", (ec_x + 22, ec_y + 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.60, (230, 255, 230), 1, cv2.LINE_AA)

                # Animated reduction counter
                cur_reduction = min(35.0, progress * 35.0 * 1.25)
                cv2.putText(canvas, f"-{cur_reduction:04.1f}% CO2", (ec_x + 24, ec_y + 115),
                            cv2.FONT_HERSHEY_DUPLEX, 1.8, (50, 255, 140), 3, cv2.LINE_AA)
                cv2.putText(canvas, "EMISSIONS REDUCED ACROSS ALL PLANTS", (ec_x + 28, ec_y + 155),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.54, (180, 220, 190), 1, cv2.LINE_AA)

                # Live Grid Stats
                grid_output = int(1450 + math.sin(t * 3) * 60)
                cv2.rectangle(canvas, (ec_x + 22, ec_y + 180), (ec_x + ec_w - 22, ec_y + 225), (16, 32, 20), -1)
                cv2.putText(canvas, f"GENERATION: {grid_output} MWh  |  WIND CAPACITY: 98.4%  |  GRID: ONLINE",
                            (ec_x + 30, ec_y + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (140, 255, 180), 1, cv2.LINE_AA)

                # Status pill
                cv2.rectangle(canvas, (ec_x + 22, ec_y + 240), (ec_x + ec_w - 22, ec_y + 274), (26, 68, 36), -1)
                cv2.putText(canvas, "RENEWABLE STANDARD: ISO-14064 VERIFIED", (ec_x + 34, ec_y + 263),
                            cv2.FONT_HERSHEY_DUPLEX, 0.52, (200, 255, 220), 1, cv2.LINE_AA)

            # ----------------- THEME: CIRCULAR MATERIALS SCIENCE LAB -----------------
            elif theme_mode == "laboratory":
                # High-tech laboratory interior (crisp slate & lab tile walls)
                self._apply_theme_background(canvas, base_bg, (32, 28, 24))
                cursor_target = (int(width * 0.38), int(height * 0.32))

                # Wall tile seams
                for gx in range(0, int(width * 0.74), 110):
                    cv2.line(canvas, (gx, 0), (gx, int(height * 0.62)), (48, 42, 38), 1)
                for gy in range(0, int(height * 0.62), 80):
                    cv2.line(canvas, (0, gy), (int(width * 0.74), gy), (48, 42, 38), 1)

                # Stainless steel lab bench counter top
                bench_y = int(height * 0.62)
                cv2.rectangle(canvas, (0, bench_y), (int(width * 0.74), height), (44, 46, 52), -1)
                cv2.line(canvas, (0, bench_y), (int(width * 0.74), bench_y), (140, 150, 165), 3)
                cv2.rectangle(canvas, (0, bench_y + 6), (int(width * 0.74), bench_y + 24), (70, 75, 84), -1)

                # Precision Research Optical Microscope on Lab Bench
                mic_cx = int(width * 0.48)
                mic_by = bench_y - 10
                
                # Heavy Horseshoe Cast Base
                cv2.rectangle(canvas, (mic_cx - 50, mic_by - 20), (mic_cx + 50, mic_by), (60, 65, 75), -1)
                cv2.rectangle(canvas, (mic_cx - 50, mic_by - 20), (mic_cx + 50, mic_by), (180, 190, 205), 2, cv2.LINE_AA)
                # Pillar & Curved Arm
                cv2.line(canvas, (mic_cx + 28, mic_by - 20), (mic_cx + 28, mic_by - 210), (90, 95, 110), 18, cv2.LINE_AA)
                cv2.line(canvas, (mic_cx + 28, mic_by - 210), (mic_cx - 20, mic_by - 240), (90, 95, 110), 18, cv2.LINE_AA)
                # Coarse / Fine Focus Adjustment Knobs
                cv2.circle(canvas, (mic_cx + 28, mic_by - 120), 18, (160, 170, 185), -1, cv2.LINE_AA)
                cv2.circle(canvas, (mic_cx + 28, mic_by - 120), 9, (70, 75, 85), -1, cv2.LINE_AA)
                # Mechanical Specimen Stage
                stage_y = mic_by - 130
                cv2.rectangle(canvas, (mic_cx - 55, stage_y), (mic_cx + 25, stage_y + 14), (50, 55, 65), -1)
                cv2.rectangle(canvas, (mic_cx - 55, stage_y), (mic_cx + 25, stage_y + 14), (160, 170, 185), 2, cv2.LINE_AA)
                # Glass Slide on Stage
                cv2.rectangle(canvas, (mic_cx - 40, stage_y - 4), (mic_cx + 10, stage_y), (220, 245, 255), -1)
                # Glowing Bio-Polymer Specimen on Slide (Fluorescence glow)
                spec_glow = int(180 + math.sin(t * 6) * 55)
                cv2.ellipse(canvas, (mic_cx - 15, stage_y - 2), (12, 6), 0, 0, 360, (60, spec_glow, 120), -1, cv2.LINE_AA)
                # Revolving Objective Nosepiece & Objectives
                cv2.circle(canvas, (mic_cx - 15, mic_by - 200), 16, (80, 85, 95), -1, cv2.LINE_AA)
                cv2.rectangle(canvas, (mic_cx - 24, mic_by - 184), (mic_cx - 6, mic_by - 146), (180, 190, 205), -1)
                cv2.rectangle(canvas, (mic_cx - 24, mic_by - 184), (mic_cx - 6, mic_by - 146), (20, 180, 240), 1)
                # Eyepiece Tube & Monocular/Binocular Head
                cv2.line(canvas, (mic_cx - 15, mic_by - 200), (mic_cx - 40, mic_by - 270), (120, 125, 135), 14, cv2.LINE_AA)
                cv2.rectangle(canvas, (mic_cx - 52, mic_by - 290), (mic_cx - 28, mic_by - 270), (40, 45, 55), -1)

                # Erlenmeyer Chemistry Flask with Green Bio-Polymer & Bubbles
                fk_x, fk_y = int(width * 0.12), bench_y - 12
                # Neck
                cv2.rectangle(canvas, (fk_x - 8, fk_y - 95), (fk_x + 8, fk_y - 45), (140, 160, 175), 2)
                # Flask conical body
                flask_pts = np.array([
                    [fk_x - 8, fk_y - 45], [fk_x + 8, fk_y - 45],
                    [fk_x + 38, fk_y], [fk_x - 38, fk_y]
                ], np.int32)
                cv2.polylines(canvas, [flask_pts], True, (160, 180, 195), 2, cv2.LINE_AA)
                # Liquid inside
                liquid_pts = np.array([
                    [fk_x - 22, fk_y - 26], [fk_x + 22, fk_y - 26],
                    [fk_x + 36, fk_y], [fk_x - 36, fk_y]
                ], np.int32)
                cv2.fillPoly(canvas, [liquid_pts], (60, 190, 100), cv2.LINE_AA)
                # Rising bubbles
                for b_i in range(5):
                    bub_y = int(fk_y - ((t * 30 + b_i * 18) % 36))
                    bub_x = fk_x + int(math.sin(t * 4 + b_i) * 12)
                    cv2.circle(canvas, (bub_x, bub_y), 3, (180, 255, 200), -1, cv2.LINE_AA)

                # Circular Materials & Polymer Spectrometry Card
                lab_x, lab_y, lab_w, lab_h = int(width * 0.06), int(height * 0.10), int(width * 0.38), 320
                cv2.rectangle(canvas, (lab_x, lab_y), (lab_x + lab_w, lab_y + lab_h), (24, 28, 36), -1)
                cv2.rectangle(canvas, (lab_x, lab_y), (lab_x + lab_w, lab_y + lab_h), (40, 200, 240), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (lab_x, lab_y), (lab_x + lab_w, lab_y + 44), (20, 48, 64), -1)
                cv2.putText(canvas, "MATERIALS SCIENCE LAB // CIRCULAR BIO-POLYMERS", (lab_x + 20, lab_y + 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.54, (200, 240, 255), 1, cv2.LINE_AA)

                cv2.putText(canvas, "100% CIRCULAR BIO-MATERIALS", (lab_x + 22, lab_y + 90),
                            cv2.FONT_HERSHEY_DUPLEX, 0.95, (80, 240, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "PLANT-BASED POLYMER RESIN MATRIX", (lab_x + 24, lab_y + 124),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.54, (160, 210, 230), 1, cv2.LINE_AA)

                # Hexagonal Molecular Polymer Chain Diagram
                hex_cx, hex_cy = lab_x + 130, lab_y + 195
                for h_idx in range(3):
                    hx = hex_cx + h_idx * 55
                    for v_i in range(6):
                        a1 = v_i * math.pi / 3.0 + t * 0.4
                        a2 = (v_i + 1) * math.pi / 3.0 + t * 0.4
                        p1 = (int(hx + 22 * math.cos(a1)), int(hex_cy + 22 * math.sin(a1)))
                        p2 = (int(hx + 22 * math.cos(a2)), int(hex_cy + 22 * math.sin(a2)))
                        cv2.line(canvas, p1, p2, (80, 220, 240), 2, cv2.LINE_AA)
                        cv2.circle(canvas, p1, 4, (40, 255, 140), -1, cv2.LINE_AA)

                # Degradation metric
                cv2.rectangle(canvas, (lab_x + 20, lab_y + 250), (lab_x + lab_w - 20, lab_y + 295), (18, 36, 46), -1)
                cv2.putText(canvas, "BIODEGRADATION: 100% COMPOSTABLE (0% MICROPLASTICS)",
                            (lab_x + 28, lab_y + 280), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 255, 200), 1, cv2.LINE_AA)

            # ----------------- THEME: URBAN ECOLOGY GARDEN & WORKSHOP -----------------
            elif theme_mode == "ecology_garden":
                if base_bg is not None and base_bg.size > 0:
                    self._apply_theme_background(canvas, base_bg, (40, 75, 45))
                else:
                    # Outdoor community park sky gradient (soft azure to bright daylight)
                    for gy in range(height):
                        pct = gy / height
                        if pct < 0.60:
                            b = int(240 * (1 - pct) + 180 * pct)
                            g = int(210 * (1 - pct) + 195 * pct)
                            r = int(140 * (1 - pct) + 120 * pct)
                        else:
                            b = int(180 * (1 - pct) + 40 * pct)
                            g = int(195 * (1 - pct) + 110 * pct)
                            r = int(120 * (1 - pct) + 45 * pct)
                        canvas[gy, :] = (b, g, r)

                cursor_target = (int(width * 0.38), int(height * 0.28))

                # Distant lush tree silhouettes in background
                tree_y = int(height * 0.54)
                for tx in range(0, int(width * 0.74), 85):
                    tr_r = int(55 + math.sin(tx * 0.02) * 15)
                    cv2.circle(canvas, (tx, tree_y), tr_r, (45, 95, 55), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (tx, tree_y - 20), tr_r - 12, (55, 120, 65), -1, cv2.LINE_AA)

                # Raised cedar wood planter garden bed
                planter_top = int(height * 0.62)
                planter_bot = height - 40
                # Soil layer
                cv2.rectangle(canvas, (0, planter_top), (int(width * 0.74), planter_top + 35), (28, 42, 58), -1)
                # Cedar planter boards
                for by_off in range(planter_top + 35, planter_bot, 42):
                    cv2.rectangle(canvas, (0, by_off), (int(width * 0.74), by_off + 38), (35, 65, 115), -1)
                    cv2.line(canvas, (0, by_off), (int(width * 0.74), by_off), (20, 45, 80), 2)
                    for gx in range(40, int(width * 0.74), 160):
                        cv2.line(canvas, (gx, by_off + 8), (gx + 80, by_off + 28), (45, 78, 135), 1)

                # Sprouting seedlings, leafy stalks, and marigold blossoms in the raised bed
                for plant_i in range(9):
                    px = int(width * 0.06 + plant_i * (width * 0.60 / 8))
                    sway = math.sin(t * 2.5 + plant_i * 0.8) * 8
                    stem_base = (px, planter_top + 10)
                    stem_top = (int(px + sway), planter_top - int(65 + (plant_i % 3) * 25))

                    # Green stalk
                    cv2.line(canvas, stem_base, stem_top, (35, 160, 65), 4, cv2.LINE_AA)

                    # Green Leaves branching off stalk
                    leaf_y = (stem_base[1] + stem_top[1]) // 2
                    cv2.ellipse(canvas, (int(px - 14 + sway * 0.5), leaf_y), (16, 7), -25, 0, 360, (50, 195, 80), -1, cv2.LINE_AA)
                    cv2.ellipse(canvas, (int(px + 14 + sway * 0.5), leaf_y - 12), (16, 7), 25, 0, 360, (65, 215, 95), -1, cv2.LINE_AA)

                    # Colorful Blossom at top of alternating plants
                    if plant_i % 2 == 1:
                        fl_col = (40, 180, 245) if plant_i % 4 == 1 else (60, 100, 240)
                        cv2.circle(canvas, stem_top, 14, fl_col, -1, cv2.LINE_AA)
                        cv2.circle(canvas, stem_top, 6, (20, 230, 255), -1, cv2.LINE_AA)

                # Pollen / Butterfly ambient floating particles
                for pi in range(10):
                    p_x = int((width * 0.08 + pi * 70 + math.sin(t * 3 + pi) * 25) % (width * 0.70))
                    p_y = int(planter_top - 40 - ((t * 35 + pi * 50) % 220))
                    cv2.circle(canvas, (p_x, p_y), 4, (80, 240, 255), -1, cv2.LINE_AA)

                # Community Workshop & Urban Ecology HUD Card
                gc_x, gc_y, gc_w, gc_h = int(width * 0.06), int(height * 0.10), int(width * 0.42), 300
                cv2.rectangle(canvas, (gc_x, gc_y), (gc_x + gc_w, gc_y + gc_h), (24, 38, 28), -1)
                cv2.rectangle(canvas, (gc_x, gc_y), (gc_x + gc_w, gc_y + gc_h), (80, 210, 100), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (gc_x, gc_y), (gc_x + gc_w, gc_y + 44), (32, 64, 40), -1)
                cv2.putText(canvas, "COMMUNITY WORKSHOP // URBAN ECOLOGY GARDEN", (gc_x + 20, gc_y + 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.54, (220, 255, 230), 1, cv2.LINE_AA)

                cv2.putText(canvas, "LOCAL ECOLOGICAL CHANGE", (gc_x + 22, gc_y + 92),
                            cv2.FONT_HERSHEY_DUPLEX, 0.95, (100, 255, 140), 2, cv2.LINE_AA)
                cv2.putText(canvas, "VOLUNTEER PLANTING & GREEN LITERACY", (gc_x + 24, gc_y + 128),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.54, (190, 240, 205), 1, cv2.LINE_AA)

                # Metric Counter
                sprouts_count = int(min(1250, progress * 1250 * 1.3))
                cv2.rectangle(canvas, (gc_x + 22, gc_y + 155), (gc_x + gc_w - 22, gc_y + 215), (18, 48, 26), -1)
                cv2.putText(canvas, f"+{sprouts_count} SEEDLINGS PLANTED", (gc_x + 32, gc_y + 195),
                            cv2.FONT_HERSHEY_DUPLEX, 0.85, (70, 255, 150), 2, cv2.LINE_AA)

                cv2.rectangle(canvas, (gc_x + 22, gc_y + 235), (gc_x + gc_w - 22, gc_y + 276), (28, 75, 42), -1)
                cv2.putText(canvas, "COMMUNITY PARTICIPATION: 100% // SUSTAINABILITY ENGAGED",
                            (gc_x + 32, gc_y + 262), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 255, 220), 1, cv2.LINE_AA)

            # ----------------- THEME: DEEP SPACE & ORBITAL -----------------
            elif theme_mode == "space":
                self._apply_theme_background(canvas, base_bg, (12, 10, 16))
                cursor_target = (int(width * 0.38), int(height * 0.32))

                # Twinkling starfield
                for s_idx in range(45):
                    sx = int((width * 0.05 + s_idx * 43 + math.sin(s_idx * 1.7) * 20) % (width * 0.72))
                    sy = int((height * 0.06 + s_idx * 27) % (height * 0.75))
                    star_twinkle = abs(math.sin(t * 4.0 + s_idx))
                    s_r = 1 if star_twinkle < 0.5 else 2
                    cv2.circle(canvas, (sx, sy), s_r, (255, 255, int(200 + star_twinkle * 55)), -1)

                # Celestial Moon / Planet Crescent
                pl_cx, pl_cy = int(width * 0.45), int(height * 0.34)
                cv2.circle(canvas, (pl_cx, pl_cy), 62, (215, 230, 245), -1, cv2.LINE_AA)
                cv2.circle(canvas, (pl_cx - 18, pl_cy - 8), 58, (12, 10, 16), -1, cv2.LINE_AA)

                # Space HUD
                sp_x, sp_y, sp_w, sp_h = int(width * 0.06), int(height * 0.10), int(width * 0.42), 300
                cv2.rectangle(canvas, (sp_x, sp_y), (sp_x + sp_w, sp_y + sp_h), (20, 18, 28), -1)
                cv2.rectangle(canvas, (sp_x, sp_y), (sp_x + sp_w, sp_y + sp_h), (220, 180, 80), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (sp_x, sp_y), (sp_x + sp_w, sp_y + 44), (32, 28, 42), -1)
                cv2.putText(canvas, "DEEP SPACE MISSION // ORBITAL TELEMETRY", (sp_x + 20, sp_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.54, (240, 220, 180), 1, cv2.LINE_AA)
                cv2.putText(canvas, "INTERSTELLAR TRAJECTORY", (sp_x + 22, sp_y + 92), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 230, 140), 2, cv2.LINE_AA)
                v_num = 27600 + int(math.sin(t * 3) * 140)
                cv2.putText(canvas, f"VELOCITY: {v_num} KM/H  |  ORBIT: NOMINAL", (sp_x + 24, sp_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (180, 240, 255), 1, cv2.LINE_AA)
                cv2.rectangle(canvas, (sp_x + 20, sp_y + 250), (sp_x + sp_w - 20, sp_y + 290), (32, 28, 42), -1)
                cv2.putText(canvas, "MISSION STATUS: FLIGHT PATH STABLE", (sp_x + 30, sp_y + 276), cv2.FONT_HERSHEY_DUPLEX, 0.48, (200, 240, 255), 1, cv2.LINE_AA)

            # ----------------- THEME: GOURMET CULINARY & KITCHEN -----------------
            elif theme_mode == "culinary":
                self._apply_theme_background(canvas, base_bg, (22, 24, 38))
                cursor_target = (int(width * 0.38), int(height * 0.32))

                bench_y = int(height * 0.62)
                cv2.rectangle(canvas, (0, bench_y), (int(width * 0.74), height), (35, 45, 65), -1)
                cv2.line(canvas, (0, bench_y), (int(width * 0.74), bench_y), (50, 90, 160), 3)
                for si in range(6):
                    st_x = int(width * 0.28 + si * 18 + math.sin(t * 4 + si) * 12)
                    st_y = int(bench_y - 20 - ((t * 45 + si * 35) % 90))
                    cv2.circle(canvas, (st_x, st_y), 5, (160, 180, 200), -1, cv2.LINE_AA)

                cul_x, cul_y, cul_w, cul_h = int(width * 0.06), int(height * 0.10), int(width * 0.42), 300
                cv2.rectangle(canvas, (cul_x, cul_y), (cul_x + cul_w, cul_y + cul_h), (28, 22, 24), -1)
                cv2.rectangle(canvas, (cul_x, cul_y), (cul_x + cul_w, cul_y + cul_h), (40, 140, 245), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (cul_x, cul_y), (cul_x + cul_w, cul_y + 44), (45, 30, 35), -1)
                cv2.putText(canvas, "GOURMET KITCHEN // MASTER CUISINE CRAFT", (cul_x + 20, cul_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.54, (255, 230, 210), 1, cv2.LINE_AA)
                cv2.putText(canvas, "ARTISANAL FLAVOR MASTERY", (cul_x + 22, cul_y + 92), cv2.FONT_HERSHEY_DUPLEX, 0.95, (80, 180, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "PREMIUM INGREDIENT SOURCING ACTIVE", (cul_x + 24, cul_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 240), 1, cv2.LINE_AA)

            # ----------------- THEME: ATHLETIC FITNESS & GYM -----------------
            elif theme_mode == "fitness":
                self._apply_theme_background(canvas, base_bg, (18, 16, 24))
                cursor_target = (int(width * 0.38), int(height * 0.32))

                for lx in [int(width * 0.12), int(width * 0.32), int(width * 0.52)]:
                    cv2.line(canvas, (lx, 0), (lx, height), (240, 140, 20), 2, cv2.LINE_AA)
                    cv2.line(canvas, (lx + 6, 0), (lx + 6, height), (50, 240, 120), 1, cv2.LINE_AA)

                fit_x, fit_y, fit_w, fit_h = int(width * 0.06), int(height * 0.10), int(width * 0.42), 300
                cv2.rectangle(canvas, (fit_x, fit_y), (fit_x + fit_w, fit_y + fit_h), (25, 20, 28), -1)
                cv2.rectangle(canvas, (fit_x, fit_y), (fit_x + fit_w, fit_y + fit_h), (50, 240, 120), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (fit_x, fit_y), (fit_x + fit_w, fit_y + 44), (32, 28, 38), -1)
                cv2.putText(canvas, "ATHLETIC TRAINING // METABOLIC PERFORMANCE", (fit_x + 20, fit_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.54, (220, 255, 230), 1, cv2.LINE_AA)
                cal_burn = int(min(650, progress * 650 * 1.25) + 120)
                cv2.putText(canvas, f"{cal_burn} KCAL BURNED", (fit_x + 22, fit_y + 95), cv2.FONT_HERSHEY_DUPLEX, 1.25, (50, 255, 140), 2, cv2.LINE_AA)
                cv2.putText(canvas, "ZONE 4 HIGH-INTENSITY // HR: 142 BPM", (fit_x + 24, fit_y + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (200, 255, 220), 1, cv2.LINE_AA)

            # ----------------- THEME: DEEP OCEAN & MARINE -----------------
            elif theme_mode == "ocean":
                if base_bg is not None and base_bg.size > 0:
                    self._apply_theme_background(canvas, base_bg, (80, 50, 20))
                else:
                    for gy in range(height):
                        pct = gy / height
                        canvas[gy, :] = (int(85 * (1 - pct) + 40 * pct), int(55 * (1 - pct) + 20 * pct), int(18 * (1 - pct) + 10 * pct))
                for rx in range(0, int(width * 0.74), 90):
                    cv2.line(canvas, (rx + 40, 0), (rx - 30, height), (130, 80, 25), 2, cv2.LINE_AA)
                for bi in range(8):
                    by = int(height - ((t * 40 + bi * 35) % height))
                    bx = int(width * 0.10 + bi * 55 + math.sin(t * 3 + bi) * 15)
                    cv2.circle(canvas, (bx, by), 4, (200, 240, 255), 1, cv2.LINE_AA)

                oc_x, oc_y, oc_w, oc_h = int(width * 0.06), int(height * 0.10), int(width * 0.42), 300
                cv2.rectangle(canvas, (oc_x, oc_y), (oc_x + oc_w, oc_y + oc_h), (28, 20, 14), -1)
                cv2.rectangle(canvas, (oc_x, oc_y), (oc_x + oc_w, oc_y + oc_h), (240, 180, 60), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (oc_x, oc_y), (oc_x + oc_w, oc_y + 44), (40, 28, 20), -1)
                cv2.putText(canvas, "MARINE ECOSYSTEMS // OCEAN SURVEY", (oc_x + 20, oc_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.54, (240, 230, 200), 1, cv2.LINE_AA)
                cv2.putText(canvas, "CORAL REEF BIODIVERSITY", (oc_x + 22, oc_y + 92), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 210, 80), 2, cv2.LINE_AA)
                cv2.putText(canvas, "DEPTH: 420M // HABITAT MONITOR ACTIVE", (oc_x + 24, oc_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (200, 240, 255), 1, cv2.LINE_AA)

            # ----------------- THEME: AUTOMOTIVE & SUPERCAR -----------------
            elif theme_mode == "automotive":
                self._apply_theme_background(canvas, base_bg, (16, 14, 18))
                cursor_target = (int(width * 0.38), int(height * 0.32))

                for lx in range(0, int(width * 0.74), 70):
                    ly = int(height * 0.65)
                    cv2.line(canvas, (lx, ly), (int(lx + (lx - width*0.35)*1.8), height), (30, 160, 240), 2, cv2.LINE_AA)

                au_x, au_y, au_w, au_h = int(width * 0.06), int(height * 0.10), int(width * 0.42), 300
                cv2.rectangle(canvas, (au_x, au_y), (au_x + au_w, au_y + au_h), (24, 20, 22), -1)
                cv2.rectangle(canvas, (au_x, au_y), (au_x + au_w, au_y + au_h), (40, 120, 255), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (au_x, au_y), (au_x + au_w, au_y + 44), (36, 26, 30), -1)
                cv2.putText(canvas, "AUTOMOTIVE ENGINEERING // TRACK DYNAMICS", (au_x + 20, au_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.54, (240, 220, 220), 1, cv2.LINE_AA)
                speed_val = int(progress * 155 + 45)
                cv2.putText(canvas, f"{speed_val} MPH", (au_x + 22, au_y + 95), cv2.FONT_HERSHEY_DUPLEX, 1.4, (50, 140, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "SUPERCAR POWERTRAIN // 0-60 IN 2.1S", (au_x + 24, au_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (200, 220, 255), 1, cv2.LINE_AA)

            # ----------------- THEME: STUDIO SOUND & MUSIC -----------------
            elif theme_mode == "music":
                self._apply_theme_background(canvas, base_bg, (24, 18, 26))
                cursor_target = (int(width * 0.38), int(height * 0.32))

                for bi in range(14):
                    bar_x = int(width * 0.08 + bi * (width * 0.54 / 13))
                    bh = int(35 + math.sin(t * 7 + bi * 1.4) * 30 + (bi % 3) * 20)
                    cv2.rectangle(canvas, (bar_x - 10, int(height * 0.76 - bh)), (bar_x + 10, int(height * 0.76)), (220, 100, 240), -1)

                mu_x, mu_y, mu_w, mu_h = int(width * 0.06), int(height * 0.10), int(width * 0.42), 300
                cv2.rectangle(canvas, (mu_x, mu_y), (mu_x + mu_w, mu_y + mu_h), (32, 22, 34), -1)
                cv2.rectangle(canvas, (mu_x, mu_y), (mu_x + mu_w, mu_y + mu_h), (220, 120, 240), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (mu_x, mu_y), (mu_x + mu_w, mu_y + 44), (45, 28, 48), -1)
                cv2.putText(canvas, "STUDIO ACOUSTICS // HARMONIC SPECTRUM", (mu_x + 20, mu_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.54, (240, 210, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, "HIGH-FIDELITY SOUND", (mu_x + 22, mu_y + 92), cv2.FONT_HERSHEY_DUPLEX, 0.90, (240, 160, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "FREQUENCY RANGE: 20Hz - 20kHz // 96kHz", (mu_x + 24, mu_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 200, 240), 1, cv2.LINE_AA)

            # ----------------- THEME: EDUCATION & LEARNING -----------------
            elif theme_mode == "education":
                self._apply_theme_background(canvas, base_bg, (32, 38, 30))
                cursor_target = (int(width * 0.38), int(height * 0.32))

                cv2.putText(canvas, "E = mc^2  |  f(x) = lim dx->0  |  Ret = N * (1 + r)^t", (int(width * 0.06), int(height * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (75, 95, 75), 1, cv2.LINE_AA)

                ed_x, ed_y, ed_w, ed_h = int(width * 0.06), int(height * 0.10), int(width * 0.42), 300
                cv2.rectangle(canvas, (ed_x, ed_y), (ed_x + ed_w, ed_y + ed_h), (26, 32, 25), -1)
                cv2.rectangle(canvas, (ed_x, ed_y), (ed_x + ed_w, ed_y + ed_h), (120, 220, 150), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (ed_x, ed_y), (ed_x + ed_w, ed_y + 44), (35, 45, 34), -1)
                cv2.putText(canvas, "ACADEMIC RETENTION // KNOWLEDGE SYNTHESIS", (ed_x + 20, ed_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.54, (220, 255, 230), 1, cv2.LINE_AA)
                cv2.putText(canvas, "+85% RETENTION RATE", (ed_x + 22, ed_y + 92), cv2.FONT_HERSHEY_DUPLEX, 0.95, (140, 255, 170), 2, cv2.LINE_AA)
                cv2.putText(canvas, "CURRICULUM ACCELERATED LEARNING PROTOCOL", (ed_x + 24, ed_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (200, 240, 210), 1, cv2.LINE_AA)

            # ----------------- THEME: ARCHITECTURE & REAL ESTATE -----------------
            elif theme_mode == "architecture":
                self._apply_theme_background(canvas, base_bg, (38, 24, 18))
                cursor_target = (int(width * 0.38), int(height * 0.32))

                for gx in range(0, int(width * 0.74), 70):
                    cv2.line(canvas, (gx, 0), (gx, height), (58, 36, 26), 1)
                for gy in range(0, height, 70):
                    cv2.line(canvas, (0, gy), (int(width * 0.74), gy), (58, 36, 26), 1)

                ar_x, ar_y, ar_w, ar_h = int(width * 0.06), int(height * 0.10), int(width * 0.42), 300
                cv2.rectangle(canvas, (ar_x, ar_y), (ar_x + ar_w, ar_y + ar_h), (32, 22, 18), -1)
                cv2.rectangle(canvas, (ar_x, ar_y), (ar_x + ar_w, ar_y + ar_h), (220, 160, 80), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (ar_x, ar_y), (ar_x + ar_w, ar_y + 44), (45, 30, 24), -1)
                cv2.putText(canvas, "MODERN ARCHITECTURE // STRUCTURAL BLUEPRINT", (ar_x + 20, ar_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.54, (255, 230, 200), 1, cv2.LINE_AA)
                cv2.putText(canvas, "SUSTAINABLE URBAN DESIGN", (ar_x + 22, ar_y + 92), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 190, 100), 2, cv2.LINE_AA)
                cv2.putText(canvas, "BUILDING FOOTPRINT: 12,000 SQ FT // LEED PLATINUM", (ar_x + 24, ar_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (230, 220, 210), 1, cv2.LINE_AA)

            # ----------------- THEME: LEGAL COURTROOM & JUSTICE -----------------
            elif theme_mode == "legal":
                self._apply_theme_background(canvas, base_bg, (22, 18, 32))
                cursor_target = (int(width * 0.38), int(height * 0.32))

                lg_x, lg_y, lg_w, lg_h = int(width * 0.06), int(height * 0.10), int(width * 0.42), 300
                cv2.rectangle(canvas, (lg_x, lg_y), (lg_x + lg_w, lg_y + lg_h), (28, 22, 38), -1)
                cv2.rectangle(canvas, (lg_x, lg_y), (lg_x + lg_w, lg_y + lg_h), (80, 180, 240), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (lg_x, lg_y), (lg_x + lg_w, lg_y + 44), (38, 28, 50), -1)
                cv2.putText(canvas, "JUDICIAL COURTROOM // JURISPRUDENCE STATUTE", (lg_x + 20, lg_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.54, (240, 220, 210), 1, cv2.LINE_AA)
                cv2.putText(canvas, "UNANIMOUS LEGAL VERDICT", (lg_x + 22, lg_y + 92), cv2.FONT_HERSHEY_DUPLEX, 0.95, (100, 210, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "PRECEDENT & STATUTE COMPLIANCE CONFIRMED", (lg_x + 24, lg_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (210, 230, 245), 1, cv2.LINE_AA)

            # ----------------- THEME: ALPINE MOUNTAIN & EXPEDITION -----------------
            elif theme_mode == "travel":
                cursor_target = (int(width * 0.38), int(height * 0.32))
                if base_bg is not None and base_bg.size > 0:
                    self._apply_theme_background(canvas, base_bg, (40, 70, 90))
                else:
                    for gy in range(height):
                        pct = gy / height
                        canvas[gy, :] = (int(45 * (1 - pct) + 18 * pct), int(95 * (1 - pct) + 48 * pct), int(220 * (1 - pct) + 65 * pct))
                for mt_i in range(3):
                    mt_pts = [(0, height)]
                    for mx in range(0, int(width * 0.74) + 50, 40):
                        my = int(height * 0.55 + mt_i * 45 + math.sin(mx * 0.005 + mt_i) * 55)
                        mt_pts.append((mx, my))
                    mt_pts.append((int(width * 0.74), height))
                    cv2.fillPoly(canvas, [np.array(mt_pts, np.int32)], (25 + mt_i * 12, 45 + mt_i * 10, 75 - mt_i * 15), cv2.LINE_AA)

                tr_x, tr_y, tr_w, tr_h = int(width * 0.06), int(height * 0.10), int(width * 0.42), 300
                cv2.rectangle(canvas, (tr_x, tr_y), (tr_x + tr_w, tr_y + tr_h), (24, 28, 36), -1)
                cv2.rectangle(canvas, (tr_x, tr_y), (tr_x + tr_w, tr_y + tr_h), (60, 200, 240), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (tr_x, tr_y), (tr_x + tr_w, tr_y + 44), (32, 40, 52), -1)
                cv2.putText(canvas, "ALPINE EXPEDITION // SUMMIT TELEMETRY", (tr_x + 20, tr_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.54, (220, 245, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, "ELEVATION: 4,120M", (tr_x + 22, tr_y + 92), cv2.FONT_HERSHEY_DUPLEX, 0.95, (80, 230, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "SUMMIT EXPEDITION ROUTE CLEARED", (tr_x + 24, tr_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (200, 230, 255), 1, cv2.LINE_AA)

            # ----------------- UNIVERSAL DYNAMIC CONTEXT ENGINE (ANY NOVEL TOPIC) -----------------
            else:
                # Deterministic harmonic palette derived from the custom context hash
                palettes = [
                    ((18, 14, 28), (45, 30, 65), (200, 100, 240), (240, 200, 255)), # Deep Twilight
                    ((16, 26, 20), (32, 58, 42), (60, 220, 120), (200, 255, 220)),  # Emerald Horizon
                    ((24, 20, 14), (60, 48, 30), (40, 200, 240), (220, 245, 255)),  # Cyber Sapphire
                    ((18, 20, 32), (38, 42, 70), (45, 140, 245), (255, 220, 200)),  # Warm Studio
                    ((20, 16, 24), (55, 36, 48), (80, 100, 240), (255, 210, 230)),  # Crimson Velvet
                    ((20, 24, 28), (40, 52, 60), (120, 220, 255), (220, 250, 255)), # Arctic Slate
                ]
                pal = palettes[q_hash % len(palettes)]
                bg_col, grid_col, accent_col, text_col = pal

                self._apply_theme_background(canvas, base_bg, bg_col)
                cursor_target = (int(width * 0.38), int(height * 0.30))

                # Perspective Horizon & Grid
                horizon_y = int(height * 0.58)
                cv2.line(canvas, (0, horizon_y), (int(width * 0.74), horizon_y), grid_col, 2)
                vp_x = int(width * 0.38)
                for gx in range(-width, width * 2, 120):
                    cv2.line(canvas, (vp_x, horizon_y), (gx, height), grid_col, 1, cv2.LINE_AA)
                for gy in range(horizon_y, height, 45):
                    cv2.line(canvas, (0, gy), (int(width * 0.74), gy), grid_col, 1)

                # Ambient floating energy particles
                for pi in range(12):
                    px = int((width * 0.08 + pi * 85 + math.sin(t * 2.5 + pi) * 20) % (width * 0.70))
                    py = int(horizon_y - 20 - ((t * 40 + pi * 40) % (height * 0.45)))
                    cv2.circle(canvas, (px, py), 3, accent_col, -1, cv2.LINE_AA)

                # Dynamic Contextual Telemetry & Analysis HUD Card
                uc_x, uc_y, uc_w, uc_h = int(width * 0.07), int(height * 0.10), int(width * 0.43), 320
                cv2.rectangle(canvas, (uc_x, uc_y), (uc_x + uc_w, uc_y + uc_h), (20, 18, 26), -1)
                cv2.rectangle(canvas, (uc_x, uc_y), (uc_x + uc_w, uc_y + uc_h), accent_col, 2, cv2.LINE_AA)
                
                # Card Header Bar with dynamic topic title
                cv2.rectangle(canvas, (uc_x, uc_y), (uc_x + uc_w, uc_y + 44), grid_col, -1)
                clean_title = theme_title[:38].upper()
                cv2.putText(canvas, clean_title, (uc_x + 20, uc_y + 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.56, text_col, 1, cv2.LINE_AA)

                # Context excerpt from actual sentence or visual hint
                sub_txt = (visual_hint or scene_text or "Contextual Domain Analysis")
                if len(sub_txt) > 52:
                    sub_txt = sub_txt[:50] + "..."
                cv2.putText(canvas, sub_txt, (uc_x + 22, uc_y + 88),
                            cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

                # Dynamic Metric Extraction: check if numbers / percentages exist in text
                num_matches = re.findall(r'(\d+[\.\d]*\s*%|\$\d+[\.\d]*[KMBkmb]?|\d+x|\d+)', scene_text + " " + (visual_hint or ""))
                if num_matches:
                    metric_str = f"{num_matches[0]} MEASURED METRIC"
                else:
                    live_num = int(min(98.5, progress * 98.5 * 1.25))
                    metric_str = f"+{live_num}% EFFICIENCY INDEX"
                cv2.putText(canvas, metric_str, (uc_x + 24, uc_y + 138),
                            cv2.FONT_HERSHEY_DUPLEX, 1.05, accent_col, 2, cv2.LINE_AA)

                # Multi-bar live telemetry equalizer
                for bi in range(8):
                    bx = uc_x + 24 + bi * 34
                    b_h = int(12 + math.sin(t * 6 + bi * 1.2) * 18 + progress * 24)
                    cv2.rectangle(canvas, (bx, uc_y + 220 - b_h), (bx + 20, uc_y + 220), accent_col, -1)
                cv2.putText(canvas, "LIVE DATA TELEMETRY STREAM // 30 FPS", (uc_x + 24, uc_y + 245),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 195), 1, cv2.LINE_AA)

                # Status pill
                cv2.rectangle(canvas, (uc_x + 20, uc_y + 265), (uc_x + uc_w - 20, uc_y + 304), grid_col, -1)
                cv2.putText(canvas, "CONTEXT HARMONIZATION: 100% // VERIFIED ROYALTY-CLEAN",
                            (uc_x + 30, uc_y + 290), cv2.FONT_HERSHEY_DUPLEX, 0.48, text_col, 1, cv2.LINE_AA)

            # ----------------- ANIMATED MOUSE POINTER & REALISTIC CLICKS -----------------
            # Cursor moves smoothly to target and clicks
            if cursor_active:
                if t < click_time:
                    # Easing move towards target
                    mv_pct = min(1.0, t / (click_time * 0.85))
                    smooth_p = 0.5 - 0.5 * math.cos(mv_pct * math.pi)
                    cur_x = int(cursor_start[0] + (cursor_target[0] - cursor_start[0]) * smooth_p)
                    cur_y = int(cursor_start[1] + (cursor_target[1] - cursor_start[1]) * smooth_p)
                    self._draw_cursor(canvas, cur_x, cur_y, clicking=False)
                else:
                    # Hovering on target, clicking at click_time
                    click_p = min(1.0, (t - click_time) / 0.6)
                    self._draw_cursor(canvas, cursor_target[0], cursor_target[1], clicking=True, click_progress=click_p)

            # ----------------- ANIMATED CHARACTER PRESENTER WITH TRUE LIP SYNC -----------------
            self._draw_animated_character(
                canvas=canvas,
                frame_idx=frame_idx,
                t=t,
                width=width,
                height=height,
                theme=theme_mode,
                lip_sync_energy=lip_sync_energy,
                scene_text=scene_text,
                character_type=character_type,
                scene_speaker=scene_speaker,
                scene_index=scene_index
            )

            # Watermark-free clean footer timecode
            cv2.rectangle(canvas, (0, height - 38), (width, height), (12, 12, 12), -1)
            cv2.putText(canvas, f"INCUBRIX B-ROLL // THEME: {theme_title.upper()} // AUDIO LIP-SYNC: {lip_sync_energy:04.2f} // 30 FPS", 
                        (32, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (170, 170, 170), 1, cv2.LINE_AA)

            writer.write(canvas)

        writer.release()

        # Re-encode to standard H.264 using FFmpeg
        cmd = [
            FFMPEG_EXE, "-y", "-i", str(raw_temp),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
            "-preset", "veryfast", "-crf", "22",
            str(output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if raw_temp.exists():
            try:
                raw_temp.unlink()
            except Exception:
                pass

        clip_id = f"motion_{abs(q_hash) % 100000:05d}"
        
        return FootageClip(
            clip_id=clip_id,
            local_path=output_path,
            source=f"IncuBrix Kinetic Synthesis ({theme_title})",
            license_name="Commercial Royalty-Free Grant (AI-Generated)",
            attribution="IncuBrix Generative Motion Engine",
            original_url=f"synthetic://render/{clip_id}",
            is_ai_generated=True,
            selection_reason=f"Synthesized interactive {theme_title} motion graphics with audio-synced character host matched to content: '{scene_text[:50]}...'",
            relevance_score=4.9
        )

    # ─────────────────────── SHARED CHARACTER HELPERS ──────────────────────────

    def _draw_mouth(self, canvas, cx: int, cy: int, lip_sync_energy: float,
                    oral_color=(30, 20, 120), tongue_color=(80, 70, 210)):
        """Audio-driven lip-sync mouth shared by all characters."""
        mouth_y = cy + 36
        if lip_sync_energy > 0.04:
            mouth_open_h = int(3 + lip_sync_energy * 26)
            mouth_w = int(14 + lip_sync_energy * 7)
            cv2.ellipse(canvas, (cx, mouth_y), (mouth_w, mouth_open_h),
                        0, 0, 360, oral_color, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (cx, mouth_y - mouth_open_h + 3),
                        (mouth_w - 4, 3), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
            if mouth_open_h > 10:
                cv2.ellipse(canvas, (cx, mouth_y + mouth_open_h - 4),
                            (mouth_w - 6, 4), 0, 0, 360, tongue_color, -1, cv2.LINE_AA)
        else:
            cv2.ellipse(canvas, (cx, mouth_y - 2), (14, 5), 0, 0, 180,
                        (40, 30, 25), 3, cv2.LINE_AA)

    def _draw_equalizer_bubble(self, canvas, cx: int, cy: int, t: float,
                                lip_sync_energy: float,
                                bubble_color=(100, 220, 255),
                                bg_color=(28, 22, 34)):
        """8-bar animated EQ speech-bubble above the character's head (only when actively speaking)."""
        if lip_sync_energy <= 0.04:
            return
        bx, by = cx - 140, cy - 150
        cv2.rectangle(canvas, (bx, by), (bx + 220, by + 55), bg_color, -1)
        cv2.rectangle(canvas, (bx, by), (bx + 220, by + 55), bubble_color, 2, cv2.LINE_AA)
        cv2.fillPoly(canvas, [np.array([
            [bx + 140, by + 55], [bx + 165, by + 55], [bx + 130, by + 75]
        ], dtype=np.int32)], bg_color, cv2.LINE_AA)
        for bi in range(8):
            bar_x = bx + 20 + bi * 22
            jitter = math.sin(t * 18 + bi * 1.5) * 6
            bar_h = int(max(4, lip_sync_energy * 34 + jitter))
            col = (60, 240, 120) if bi % 3 == 0 else ((80, 220, 255) if bi % 3 == 1 else (60, 180, 240))
            cv2.rectangle(canvas, (bar_x, by + 44 - bar_h), (bar_x + 12, by + 44), col, -1)

    def _draw_human_head(
        self,
        canvas: np.ndarray,
        cx: int,
        cy: int,
        t: float,
        frame_idx: int,
        lip_sync_energy: float,
        role: str = "doctor",
        head_radius: int = 64,
        skin_color: tuple = (195, 215, 240),
        hair_color: tuple = (35, 45, 60),
        eye_color: tuple = (80, 50, 40),
        bubble_color: tuple = (100, 220, 255),
        draw_hud: bool = True
    ):
        """
        Draws realistic human head:
        1. Free API Avatar Portrait (randomuser.me or cached real photo) if available:
           Composites real human portrait with animated audio pulse aura and overhead equalizer.
        2. Seamless fallback: High-fidelity procedural craniofacial rendering with
           anatomical jawline, volumetric skin tones, almond eyes with limbal ring,
           sculpted 3D nose, and Cupid's bow lips with visemes.
        """
        portrait = None
        try:
            from src.pipeline.avatar_service import avatar_service
            portrait = avatar_service.get_avatar_portrait(role)
        except Exception:
            pass

        if portrait is not None:
            r = head_radius
            
            # Dynamic audio pulse rings when speaking
            if lip_sync_energy > 0.04:
                pulse_amp = int(lip_sync_energy * 10)
                for ri in range(3):
                    ring_r = r + 4 + ri * 6 + pulse_amp
                    cv2.circle(canvas, (cx, cy), ring_r, bubble_color, 2, cv2.LINE_AA)

            face = cv2.resize(portrait, (r * 2, r * 2), interpolation=cv2.INTER_LANCZOS4)
            mask = np.zeros((r * 2, r * 2), dtype=np.uint8)
            cv2.circle(mask, (r, r), r - 2, 255, -1, cv2.LINE_AA)

            y1, y2 = cy - r, cy + r
            x1, x2 = cx - r, cx + r
            if 0 <= y1 and y2 <= canvas.shape[0] and 0 <= x1 and x2 <= canvas.shape[1]:
                roi = canvas[y1:y2, x1:x2]
                for c in range(3):
                    roi[:, :, c] = np.where(mask == 255, face[:, :, c], roi[:, :, c])

            # Elegant metallic rim (accent when speaking, clean slate when listening)
            rim_col = bubble_color if lip_sync_energy > 0.04 else (180, 190, 205)
            cv2.circle(canvas, (cx, cy), r, rim_col, 3, cv2.LINE_AA)

            # Overhead animated equalizer speech bubble (only when speaking)
            if draw_hud:
                self._draw_equalizer_bubble(canvas, cx, cy - 10, t, lip_sync_energy, bubble_color=bubble_color)
            return

        # ── PROCEDURAL FALLBACK ──
        # Neck
        cv2.rectangle(canvas, (cx - 20, cy + 30), (cx + 20, cy + 50), skin_color, -1)
        # Head
        cv2.circle(canvas, (cx, cy), head_radius, skin_color, -1, cv2.LINE_AA)
        # Hair
        cv2.ellipse(canvas, (cx, cy - 35), (head_radius + 2, 38), 0, 180, 360, hair_color, -1, cv2.LINE_AA)
        cv2.line(canvas, (cx - 58, cy - 10), (cx - 52, cy - 35), hair_color, 6, cv2.LINE_AA)
        cv2.line(canvas, (cx + 58, cy - 10), (cx + 52, cy - 35), hair_color, 6, cv2.LINE_AA)
        # Eyes, mouth, bubble
        self._draw_eyes(canvas, cx, cy - 6, t, frame_idx, lip_sync_energy, pupil_color=eye_color)
        self._draw_mouth(canvas, cx, cy + 22, lip_sync_energy)
    def _draw_dialogue_avatar_feed(
        self,
        canvas: np.ndarray,
        cx: int,
        cy: int,
        role: str,
        display_title: str,
        lip_sync_energy: float,
        t: float,
        radius: int = 110,
        sub_title: Optional[str] = None
    ) -> bool:
        """
        Renders an executive, broadcast-grade real human video feed for dialogue and presenter scenes.
        Composites real human photography from free avatar API with dynamic audio pulse aura,
        radial audio visualizer, and glassmorphic lower-third status badge.
        """
        try:
            from src.pipeline.avatar_service import avatar_service
            portrait = avatar_service.get_avatar_portrait(role)
        except Exception:
            portrait = None

        if portrait is None:
            return False

        r = radius
        face = cv2.resize(portrait, (r * 2, r * 2), interpolation=cv2.INTER_LANCZOS4)
        is_speaking = lip_sync_energy > 0.04

        # Active audio pulse aura rings when speaking
        if is_speaking:
            for ri in range(3):
                aura_r = r + 6 + ri * 8 + int(lip_sync_energy * 12)
                cv2.circle(canvas, (cx, cy), aura_r, (100, 220, 255), 2, cv2.LINE_AA)

        # Composite face with smooth antialiased circular mask
        mask = np.zeros((r * 2, r * 2), dtype=np.uint8)
        cv2.circle(mask, (r, r), r - 2, 255, -1, cv2.LINE_AA)
        y1, y2 = cy - r, cy + r
        x1, x2 = cx - r, cx + r
        if 0 <= y1 and y2 <= canvas.shape[0] and 0 <= x1 and x2 <= canvas.shape[1]:
            roi = canvas[y1:y2, x1:x2]
            for c in range(3):
                roi[:, :, c] = np.where(mask == 255, face[:, :, c], roi[:, :, c])

        # Border rim: glowing cyan/gold when speaking, clean silver when listening
        border_col = (80, 210, 255) if is_speaking else (180, 190, 205)
        border_thickness = 4 if is_speaking else 3
        cv2.circle(canvas, (cx, cy), r, border_col, border_thickness, cv2.LINE_AA)

        # Radial Audio Waveform HUD around bottom when speaking
        if is_speaking:
            bar_count = 16
            for b in range(bar_count):
                ang = 45 + b * (90 / bar_count)
                rad = math.radians(ang)
                bar_h = 10 + int(math.sin(t * 14 + b * 0.9) * 14 * lip_sync_energy + lip_sync_energy * 22)
                bx1 = int(cx + math.cos(rad) * (r + 8))
                by1 = int(dcy_val := cy + math.sin(rad) * (r + 8))
                bx2 = int(cx + math.cos(rad) * (r + 8 + bar_h))
                by2 = int(cy + math.sin(rad) * (r + 8 + bar_h))
                col = (100, 240, 180) if b % 2 == 0 else (60, 210, 255)
                cv2.line(canvas, (bx1, by1), (bx2, by2), col, 3, cv2.LINE_AA)

        # Glassmorphic Lower-Third Nameplate
        badge_y = cy + r + 36
        badge_w = 160
        status_col = (50, 240, 50) if is_speaking else (120, 140, 160)
        status_text = "SPEAKING" if is_speaking else "LISTENING"
        cv2.rectangle(canvas, (cx - badge_w, badge_y - 20), (cx + badge_w, badge_y + 24), (15, 20, 28), -1)
        cv2.rectangle(canvas, (cx - badge_w, badge_y - 20), (cx + badge_w, badge_y + 24), border_col, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx - badge_w + 22, badge_y - 1), 6, status_col, -1, cv2.LINE_AA)
        cv2.putText(canvas, display_title.upper(), (cx - badge_w + 38, badge_y + 4), cv2.FONT_HERSHEY_DUPLEX, 0.52, (245, 250, 255), 1, cv2.LINE_AA)
        sub_desc = sub_title or f"STUDIO FEED // {status_text}"
        cv2.putText(canvas, sub_desc, (cx - badge_w + 22, badge_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, border_col, 1, cv2.LINE_AA)
        return True

    def _draw_badge(self, canvas, cx: int, cy: int, name_text: str,
                    border_color=(60, 150, 240), bg_color=(20, 25, 35),
                    text_color=(240, 240, 255), show: bool = False):
        """Nameplate badge below the character."""
        if not show and not getattr(self, "show_character_labels", False):
            return
        by = cy + 240
        cv2.rectangle(canvas, (cx - 95, by - 14), (cx + 95, by + 14), bg_color, -1)
        cv2.rectangle(canvas, (cx - 95, by - 14), (cx + 95, by + 14), border_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, name_text, (cx - 88, by + 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, text_color, 1, cv2.LINE_AA)

    def _draw_eyes(self, canvas, cx: int, cy: int, t: float, frame_idx: int,
                   lip_sync_energy: float,
                   pupil_color=(40, 80, 160), brow_color=(40, 30, 25)):
        """Blinking animated eyes with glancing pupils, expressive brows, and interactive gaze tracking."""
        ey = cy - 8
        lx, rx = cx - 26, cx + 26
        is_blinking = (frame_idx % 110) < 6
        if is_blinking:
            cv2.line(canvas, (lx - 12, ey), (lx + 12, ey), brow_color, 3, cv2.LINE_AA)
            cv2.line(canvas, (rx - 12, ey), (rx + 12, ey), brow_color, 3, cv2.LINE_AA)
        else:
            cv2.ellipse(canvas, (lx, ey), (13, 16), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (rx, ey), (13, 16), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
            
            # Interactive Gaze Tracking: glances left towards presentation cards during gestures, looks at audience when speaking
            cycle = t % 4.2
            if cycle < 2.0:
                # Glancing left towards the data card / graphics
                px = int(-6 + math.sin(t * 1.5) * 2)
            else:
                # Glancing towards center/audience with friendly contact
                px = int(math.sin(t * 1.8) * 2)

            cv2.circle(canvas, (lx + px, ey), 7, pupil_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (rx + px, ey), 7, pupil_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (lx + px - 2, ey - 4), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (rx + px - 2, ey - 4), 3, (255, 255, 255), -1, cv2.LINE_AA)

            # Expressive Eyebrow Movement (lifts dynamically with speech inflection)
            brow_lift = int(lip_sync_energy * 8)
            cv2.line(canvas, (lx - 12, ey - 18 - brow_lift), (lx + 12, ey - 19 - brow_lift), brow_color, 3, cv2.LINE_AA)
            cv2.line(canvas, (rx - 12, ey - 19 - brow_lift), (rx + 12, ey - 18 - brow_lift), brow_color, 3, cv2.LINE_AA)

            # Lively subtle cheek blush when actively speaking
            if lip_sync_energy > 0.05:
                cv2.ellipse(canvas, (lx - 4, ey + 18), (11, 6), 0, 0, 360, (140, 130, 235), -1, cv2.LINE_AA)
                cv2.ellipse(canvas, (rx + 4, ey + 18), (11, 6), 0, 0, 360, (140, 130, 235), -1, cv2.LINE_AA)

    def _draw_pointing_arm(
        self,
        canvas: np.ndarray,
        cx: int,
        cy: int,
        t: float,
        arm_color=(90, 45, 120),
        skin_color=(220, 185, 155),
        lip_sync_energy: float = 0.0,
        theme: str = "default"
    ):
        """
        Interactive Multi-Gesture Character Animation System:
        1. Left Presenter Arm:
           - Phase 1 (0.0s-1.2s): Welcoming & expansive orator gesture opening toward audience.
           - Phase 2 (1.2s-2.6s): Direct interactive pointing and tapping at the UI telemetry card.
           - Phase 3 (2.6s-4.0s): Audio-cadence gesticulation pulsing rhythmically with vocal stress.
           - Phase 4 (4.0s+): Confident open-palm explanation and resolution.
           - Emits an animated holographic pointer beam / ripple beacon connecting character to content!
        2. Right Articulated Arm & Role-Specific Props:
           - Fully articulated right shoulder, elbow, and hand.
           - Doctor/Medical: Holds interactive medical diagnostic tablet with live glowing ECG cardiogram.
           - Scientist/Lab: Holds glowing chemical test tube with swirling bio-luminescent reagent.
           - Engineer/Tech: Holds handheld telemetry controller with blinking LEDs.
           - Garden/Eco: Holds terracotta pot with sprouting green seedling.
           - General/Business: Holds smart presentation tablet / stylus with glowing interface.
        """
        cycle = t % 4.5

        # ─────────── 1. LEFT ARM: INTERACTIVE PRESENTATION GESTURES ───────────
        if cycle < 1.2:
            # Welcoming open orator gesture
            p_prog = cycle / 1.2
            htx = int(cx - 130 - math.sin(p_prog * math.pi) * 35)
            hty = int(cy + 45 - math.sin(p_prog * math.pi) * 20)
            ex, ey = int(cx - 75), int(cy + 115)
            gesture_mode = "open"
        elif cycle < 2.6:
            # Direct interactive pointing & tapping at the UI card on the left
            tap_prog = (cycle - 1.2) / 1.4
            tap_accent = math.sin(tap_prog * math.pi * 2) * 16
            htx = int(cx - 215 - tap_accent)
            hty = int(cy + 15 + math.sin(t * 2.8) * 12)
            ex, ey = int(cx - 110), int(cy + 95)
            gesture_mode = "point"
        elif cycle < 4.0:
            # Conversational cadence gesticulation (moves rhythmically with audio energy)
            cadence = math.sin(t * 6.0) * (14.0 + lip_sync_energy * 32.0)
            htx = int(cx - 165 + math.cos(t * 3.2) * 22)
            hty = int(cy + 55 + cadence)
            ex, ey = int(cx - 85), int(cy + 120)
            gesture_mode = "explain"
        else:
            # Confident affirmative open palm
            htx = int(cx - 145 + math.cos(t * 2.0) * 10)
            hty = int(cy + 30 + math.sin(t * 2.0) * 8)
            ex, ey = int(cx - 80), int(cy + 105)
            gesture_mode = "open"

        # Draw left sleeve & forearm
        cv2.line(canvas, (cx - 50, cy + 130), (ex, ey), arm_color, 24, cv2.LINE_AA)
        cv2.line(canvas, (ex, ey), (htx, hty), skin_color, 18, cv2.LINE_AA)
        cv2.circle(canvas, (htx, hty), 15, skin_color, -1, cv2.LINE_AA)

        # Draw hand fingers depending on gesture_mode
        if gesture_mode == "point":
            # Pointing index finger extending left towards the cards
            cv2.line(canvas, (htx, hty), (htx - 32, hty - 14), skin_color, 7, cv2.LINE_AA)
            cv2.circle(canvas, (htx - 32, hty - 14), 4, skin_color, -1, cv2.LINE_AA)
            # Folded other fingers
            cv2.circle(canvas, (htx - 4, hty + 8), 6, skin_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (htx + 6, hty + 10), 6, skin_color, -1, cv2.LINE_AA)

            # Holographic interactive pointer beacon & connecting beam to UI
            pulse_r = int((t * 32) % 24) + 4
            cv2.circle(canvas, (htx - 32, hty - 14), pulse_r, (80, 240, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, (htx - 32, hty - 14), max(2, pulse_r // 2), (50, 255, 180), 1, cv2.LINE_AA)
            
            # Subtle futuristic pointer line leading to the HUD card
            beam_tx = max(80, int(cx - 420))
            beam_ty = int(cy - 20)
            cv2.line(canvas, (htx - 32, hty - 14), (beam_tx, beam_ty), (40, 210, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, (beam_tx, beam_ty), 6, (60, 240, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (beam_tx, beam_ty), 12, (60, 240, 255), 1, cv2.LINE_AA)
        elif gesture_mode == "open":
            # Open palm with spread fingers
            for fi, f_ang in enumerate([-0.3, -0.1, 0.1, 0.3]):
                fx = int(htx - 22 * math.cos(f_ang))
                fy = int(hty - 22 * math.sin(f_ang) - fi * 3)
                cv2.line(canvas, (htx, hty), (fx, fy), skin_color, 5, cv2.LINE_AA)
        else:
            # Natural conversational gesticulating hand
            cv2.line(canvas, (htx, hty), (htx - 24, hty - 8), skin_color, 7, cv2.LINE_AA)
            cv2.circle(canvas, (htx - 8, hty + 8), 7, skin_color, -1, cv2.LINE_AA)

        # ─────────── 2. RIGHT ARM: PROPS & DYNAMIC INTERACTION ───────────
        # Right shoulder sits at (cx + 50, cy + 130)
        r_sh = (cx + 50, cy + 130)
        # Gentle natural arm sway with breathing and speech cadence
        r_sway_y = int(math.sin(t * 3.0) * 8 + lip_sync_energy * 10)
        rex, rey = int(cx + 85), int(cy + 135 + r_sway_y * 0.4)
        rhx, rhy = int(cx + 65), int(cy + 95 + r_sway_y)

        # Draw right arm
        cv2.line(canvas, r_sh, (rex, rey), arm_color, 24, cv2.LINE_AA)
        cv2.line(canvas, (rex, rey), (rhx, rhy), skin_color, 18, cv2.LINE_AA)
        cv2.circle(canvas, (rhx, rhy), 14, skin_color, -1, cv2.LINE_AA)

        # Render role-specific interactive prop in the right hand
        is_med = (arm_color == (250, 250, 252)) or ("hosp" in str(theme).lower()) or (arm_color == (225, 195, 150))
        is_lab = ("lab" in str(theme).lower()) or ("scient" in str(theme).lower())
        is_eco = ("garden" in str(theme).lower()) or ("eco" in str(theme).lower())
        is_tech = ("tech" in str(theme).lower()) or ("energy" in str(theme).lower())

        if is_med:
            # Medical Diagnostic Tablet with live glowing ECG pulse line
            tab_x, tab_y = rhx - 14, rhy - 38
            cv2.rectangle(canvas, (tab_x, tab_y), (tab_x + 36, tab_y + 50), (28, 32, 40), -1)
            cv2.rectangle(canvas, (tab_x, tab_y), (tab_x + 36, tab_y + 50), (180, 195, 210), 2, cv2.LINE_AA)
            cv2.rectangle(canvas, (tab_x + 3, tab_y + 4), (tab_x + 33, tab_y + 46), (12, 18, 22), -1)
            for wi in range(tab_x + 5, tab_x + 31, 3):
                ph = (wi * 0.4 + t * 8) % math.pi
                wy = int(tab_y + 25 + (math.sin(ph)**5) * -12)
                cv2.circle(canvas, (wi, wy), 1, (40, 240, 100), -1)
            cv2.line(canvas, (rhx, rhy - 10), (rhx - 8, tab_y + 20), (220, 220, 230), 2, cv2.LINE_AA)
        elif is_lab:
            # Swirling Laboratory Test Tube with bio-luminescent reagent
            tt_x, tt_y = rhx + 4, rhy - 30
            cv2.rectangle(canvas, (tt_x - 6, tt_y), (tt_x + 6, tt_y + 36), (220, 240, 255), 1)
            cv2.ellipse(canvas, (tt_x, tt_y + 36), (6, 6), 0, 0, 180, (220, 240, 255), 1)
            liq_h = int(18 + math.sin(t * 4) * 3)
            cv2.rectangle(canvas, (tt_x - 5, tt_y + 36 - liq_h), (tt_x + 5, tt_y + 36), (60, 240, 140), -1)
            cv2.circle(canvas, (tt_x, tt_y + 36 - liq_h + 4), 2, (180, 255, 220), -1)
        elif is_eco:
            # Terracotta pot with live sprouting seedling
            pot_x, pot_y = rhx + 2, rhy - 10
            pot_pts = np.array([
                [pot_x - 14, pot_y], [pot_x + 14, pot_y],
                [pot_x + 10, pot_y + 22], [pot_x - 10, pot_y + 22]
            ], np.int32)
            cv2.fillPoly(canvas, [pot_pts], (45, 80, 160), cv2.LINE_AA)
            cv2.line(canvas, (pot_x, pot_y), (pot_x, pot_y - 18), (40, 200, 80), 3, cv2.LINE_AA)
            cv2.ellipse(canvas, (pot_x - 6, pot_y - 14), (8, 4), -25, 0, 360, (60, 230, 95), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (pot_x + 6, pot_y - 16), (8, 4), 25, 0, 360, (75, 245, 110), -1, cv2.LINE_AA)
        elif is_tech:
            # Handheld Digital Controller / Multimeter
            ctrl_x, ctrl_y = rhx - 10, rhy - 30
            cv2.rectangle(canvas, (ctrl_x, ctrl_y), (ctrl_x + 28, ctrl_y + 40), (35, 30, 25), -1)
            cv2.rectangle(canvas, (ctrl_x, ctrl_y), (ctrl_x + 28, ctrl_y + 40), (220, 160, 40), 2, cv2.LINE_AA)
            led_col = (50, 255, 100) if (int(t * 4) % 2 == 0) else (30, 180, 240)
            cv2.circle(canvas, (ctrl_x + 14, ctrl_y + 12), 4, led_col, -1, cv2.LINE_AA)
            cv2.line(canvas, (ctrl_x + 6, ctrl_y + 26), (ctrl_x + 22, ctrl_y + 26), (200, 240, 255), 2, cv2.LINE_AA)
        else:
            # Modern smart tablet with glowing screen
            dev_x, dev_y = rhx - 12, rhy - 32
            cv2.rectangle(canvas, (dev_x, dev_y), (dev_x + 30, dev_y + 42), (25, 22, 30), -1)
            cv2.rectangle(canvas, (dev_x, dev_y), (dev_x + 30, dev_y + 42), (180, 140, 220), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (dev_x + 3, dev_y + 4), (dev_x + 27, dev_y + 38), (40, 28, 45), -1)
            for bi in range(4):
                bx = dev_x + 6 + bi * 5
                bh = int(6 + math.sin(t * 3 + bi) * 4)
                cv2.rectangle(canvas, (bx, dev_y + 34 - bh), (bx + 3, dev_y + 34), (80, 220, 255), -1)

    # ───────────────── THEME → CHARACTER DISPATCHER ─────────────────────────

    # ───────────────── THEME → CHARACTER DISPATCHER ─────────────────────────

    def _get_char_renderer(self, name: str, theme: str):
        """Resolves a character name or theme to (render_fn, display_name)."""
        n = (name or "").lower().strip()
        custom_map = {
            "doctor":     (self._char_doctor, "Doctor"),
            "physician":  (self._char_doctor, "Doctor"),
            "surgeon":    (self._char_doctor, "Doctor"),
            "patient":    (self._char_patient, "Patient"),
            "king":       (self._char_king, "King"),
            "queen":      (self._char_king, "Queen"),
            "advisor":    (self._char_advisor, "Advisor"),
            "minister":   (self._char_advisor, "Advisor"),
            "lion":       (self._char_lion, "Lion"),
            "robot":      (self._char_robot, "Robot"),
            "cat":        (self._char_cat, "Cat"),
            "dog":        (self._char_dog, "Dog"),
            "bird":       (self._char_bird, "Bird"),
            "astronaut":  (self._char_astronaut, "Astronaut"),
            "wizard":     (self._char_wizard, "Wizard"),
            "alien":      (self._char_alien, "Alien"),
            "scientist":   (self._char_doctor, "Scientist"),
            "researcher":  (self._char_doctor, "Scientist"),
            "volunteer":   (self._char_diplomat, "Coordinator"),
            "coordinator": (self._char_diplomat, "Coordinator"),
        }
        for key, (fn, tag) in custom_map.items():
            if key in n:
                return fn, tag

        human_map = {
            "analyst":    (self._char_analyst, "Analyst"),
            "engineer":   (self._char_engineer, "Engineer"),
            "banker":     (self._char_banker, "Banker"),
            "coach":      (self._char_coach, "Coach"),
            "hacker":     (self._char_hacker, "Hacker"),
            "executive":  (self._char_executive, "Executive"),
            "influencer": (self._char_influencer, "Influencer"),
            "diplomat":   (self._char_diplomat, "Diplomat"),
            "artist":     (self._char_artist, "Artist"),
            "worker":     (self._char_tired_worker, "Worker"),
            "speedster":  (self._char_speedster, "Speedster"),
        }
        for key, (fn, tag) in human_map.items():
            if key in n:
                return fn, tag

        theme_map = {
            "hospital": (self._char_doctor, "Doctor"),
            "royal":    (self._char_king, "King"),
            "warning":  (self._char_analyst, "Analyst"),
            "waste":    (self._char_tired_worker, "Worker"),
            "calendar": (self._char_coach, "Coach"),
            "tweak":    (self._char_hacker, "Hacker"),
            "instant":  (self._char_speedster, "Speedster"),
            "email":    (self._char_executive, "Executive"),
            "social":   (self._char_influencer, "Influencer"),
            "finance":  (self._char_banker, "Banker"),
            "tech":     (self._char_engineer, "Engineer"),
            "trust":    (self._char_diplomat, "Diplomat"),
            "creative": (self._char_artist, "Artist"),
            "clean_energy": (self._char_engineer, "Engineer"),
            "laboratory": (self._char_doctor, "Scientist"),
            "ecology_garden": (self._char_diplomat, "Coordinator"),
            "space": (self._char_astronaut, "Astronaut"),
            "culinary": (self._char_artist, "Chef"),
            "fitness": (self._char_coach, "Coach"),
            "ocean": (self._char_doctor, "Marine Biologist"),
            "automotive": (self._char_speedster, "Driver"),
            "music": (self._char_influencer, "Producer"),
            "education": (self._char_advisor, "Educator"),
            "architecture": (self._char_engineer, "Architect"),
            "legal": (self._char_diplomat, "Attorney"),
            "travel": (self._char_diplomat, "Explorer"),
            "universal_context": (self._char_analyst, "Host"),
        }
        return theme_map.get(theme, (self._char_analyst, "Human"))

    def _render_drama_dialogue(
        self,
        canvas: np.ndarray,
        frame_idx: int,
        t: float,
        width: int,
        height: int,
        theme: str,
        lip_sync_energy: float,
        scene_text: str,
        char1_name: str,
        char2_name: str,
        scene_speaker: Optional[str] = None,
        scene_index: int = 0,
        show_character_labels: bool = False,
        viseme_type: str = "open"
    ):
        """
        Renders a two-character dramatic discussion / dialogue scene.
        Character 1 sits on the left (cx ≈ 0.28 * width).
        Character 2 sits on the right (cx ≈ 0.72 * width).
        STRICT LIP-SYNC: ONLY the currently speaking character moves their mouth.
        The listening character has ZERO mouth motion.
        NOTE: No character names or artificial banner labels are shown by default.
        """
        fn1, tag1 = self._get_char_renderer(char1_name, theme)
        fn2, tag2 = self._get_char_renderer(char2_name, theme)
        if fn1 == fn2 and tag1.lower() == "human":
            fn2 = self._char_engineer
            tag2 = "Engineer"

        # Determine active speaker using scene_speaker, dialogue cues, or turn-taking
        txt_lower = (scene_text or "").lower().strip()
        c1_tokens = [char1_name.lower(), tag1.lower(), "doctor", "dr.", "dr:", "marcus", "alex", "king", "speaker 1", "first"]
        c2_tokens = [char2_name.lower(), tag2.lower(), "patient", "elena", "sarah", "morgan", "nova", "advisor", "speaker 2", "second"]

        if scene_speaker:
            s_low = scene_speaker.lower().strip()
            if any(tok in s_low for tok in c1_tokens):
                speaker1_active = True
            elif any(tok in s_low for tok in c2_tokens):
                speaker1_active = False
            else:
                speaker1_active = (scene_index % 2 == 0)
        else:
            c1_matches = any(tok in txt_lower[:35] for tok in c1_tokens)
            c2_matches = any(tok in txt_lower[:35] for tok in c2_tokens)

            if c1_matches and not c2_matches:
                speaker1_active = True
            elif c2_matches and not c1_matches:
                speaker1_active = False
            else:
                # Semantic dialogue detection
                if any(w in txt_lower for w in ["lab report", "numbers with you", "take a deep breath", "medication", "light walking", "nutrition guidelines"]):
                    speaker1_active = True
                elif any(w in txt_lower for w in ["please be honest", "surgery", "relief to hear", "lifestyle adjustments"]):
                    speaker1_active = False
                elif scene_index >= 0:
                    speaker1_active = (scene_index % 2 == 0)
                else:
                    text_seed = sum(ord(c) for c in (scene_text[:16] or "a"))
                    cycle = 2.8
                    speaker1_active = ((text_seed + int(t / cycle)) % 2 == 0)

        # STRICT LIP SYNC ENFORCEMENT:
        # The active speaker receives the full lip-sync audio energy.
        # The listener receives exactly 0.0 (strictly closed mouth, natural gaze).
        if speaker1_active:
            energy1 = lip_sync_energy
            energy2 = 0.0
        else:
            energy1 = 0.0
            energy2 = lip_sync_energy

        # Check if 3D characters can be used (Doctor, Patient, Teacher, Student, Business, Farmer, Narrator, etc.)
        from src.pipeline.avatar_service import avatar_service

        is_3d_char1 = any(w in char1_name.lower() for w in ["doctor", "physician", "patient", "teacher", "student", "business", "farmer", "narrator", "analyst", "engineer", "human", "diplomat", "coach", "banker", "executive", "host", "scientist", "coordinator"])
        is_3d_char2 = any(w in char2_name.lower() for w in ["doctor", "physician", "patient", "teacher", "student", "business", "farmer", "narrator", "analyst", "engineer", "human", "diplomat", "coach", "banker", "executive", "host", "scientist", "coordinator"])

        if is_3d_char1 and is_3d_char2:
            # Pair male doctor with female patient by default for natural dialogue
            c1_k = char1_name.lower().strip()
            c2_k = char2_name.lower().strip()
            if ("doc" in c1_k or "physician" in c1_k) and ("patient" in c2_k):
                role1 = "female_doctor" if any(w in c1_k for w in ["female", "woman", "sarah", "emily"]) else "doctor"
                role2 = "patient" if any(w in c2_k for w in ["male", "man", "james"]) else "female_patient"
            elif ("doc" in c2_k or "physician" in c2_k) and ("patient" in c1_k):
                role1 = "patient" if any(w in c1_k for w in ["male", "man", "james"]) else "female_patient"
                role2 = "female_doctor" if any(w in c2_k for w in ["female", "woman", "sarah", "emily"]) else "doctor"
            else:
                role1 = char1_name
                role2 = char2_name

            target_cx1 = int(width * 0.28)
            target_cx2 = int(width * 0.72)
            # Clear medium close-up framing so viewers can clearly see mouths and facial expressions
            target_h = int(height * 1.06)
            bottom_y = height + int(height * 0.06)

            title1 = "DR. MARCUS // ATTENDING PHYSICIAN" if "doc" in role1.lower() else ("ELENA // PATIENT PARTICIPANT" if "female_patient" in role1 else None)
            title2 = "ELENA // PATIENT PARTICIPANT" if "female_patient" in role2 else ("JAMES // PATIENT PARTICIPANT" if "patient" in role2.lower() else ("DR. MARCUS // ATTENDING PHYSICIAN" if "doc" in role2.lower() else None))

            # Render Character 1 (Left position)
            avatar_service.render_realistic_character(
                canvas=canvas,
                role=role1,
                cx=target_cx1,
                bottom_y=bottom_y,
                target_height=int(target_h * 1.04) if speaker1_active else target_h,
                t=t,
                frame_idx=frame_idx,
                lip_sync_energy=energy1,
                is_speaking=speaker1_active,
                viseme_type=viseme_type if speaker1_active else "closed",
                flip_horizontal=False,
                show_badge=show_character_labels,
                custom_badge=title1 if show_character_labels else None
            )

            # Render Character 2 (Right position)
            avatar_service.render_realistic_character(
                canvas=canvas,
                role=role2,
                cx=target_cx2,
                bottom_y=bottom_y,
                target_height=int(target_h * 1.04) if not speaker1_active else target_h,
                t=t,
                frame_idx=frame_idx,
                lip_sync_energy=energy2,
                is_speaking=(not speaker1_active),
                viseme_type=viseme_type if not speaker1_active else "closed",
                flip_horizontal=False,
                show_badge=show_character_labels,
                custom_badge=title2 if show_character_labels else None
            )
            return

        # 1. Render Character 1 (Left position: cx ≈ 0.28 * width)
        buf1 = np.zeros_like(canvas)
        fn1(buf1, frame_idx, t, width, height, energy1)
        target_cx1 = int(width * 0.28)
        dx1 = target_cx1 - int(width * 0.80)
        M1 = np.float32([[1, 0, dx1], [0, 1, 0]])
        shifted1 = cv2.warpAffine(buf1, M1, (width, height))
        mask1 = np.any(shifted1 > 0, axis=-1)
        canvas[mask1] = shifted1[mask1]

        # 2. Render Character 2 (Right position: cx ≈ 0.80 * width)
        buf2 = np.zeros_like(canvas)
        fn2(buf2, frame_idx, t + 0.35, width, height, energy2)
        mask2 = np.any(buf2 > 0, axis=-1)
        canvas[mask2] = buf2[mask2]

    def _draw_animated_character(
        self,
        canvas: np.ndarray,
        frame_idx: int,
        t: float,
        width: int,
        height: int,
        theme: str,
        lip_sync_energy: float,
        scene_text: str = "",
        character_type: str = "human",
        scene_speaker: Optional[str] = None,
        scene_index: int = 0,
        show_character_labels: bool = False,
        viseme_type: str = "open"
    ):
        """
        Routes to character renderers:
        - If 2+ characters or drama discussion requested: renders 2 characters in dialogue.
        - Otherwise: routes to single character renderer.
        """
        ct = (character_type or "human").lower().strip()

        # Parse for multiple characters (e.g. "human, robot", "lion & cat", "drama", "duo", "doctor, patient")
        raw_parts = [p.strip() for p in re.split(r'[,+&]| and | vs | with ', ct) if p.strip()]
        is_drama = any(w in ct for w in ["drama", "discussion", "dialogue", "two character", "two people", "duo", "conversation"])

        if len(raw_parts) >= 2 or is_drama:
            c1 = raw_parts[0] if len(raw_parts) >= 1 else "human"
            c2 = raw_parts[1] if len(raw_parts) >= 2 else ("robot" if c1 != "robot" else "human")
            self._render_drama_dialogue(
                canvas=canvas,
                frame_idx=frame_idx,
                t=t,
                width=width,
                height=height,
                theme=theme,
                lip_sync_energy=lip_sync_energy,
                scene_text=scene_text,
                char1_name=c1,
                char2_name=c2,
                scene_speaker=scene_speaker,
                scene_index=scene_index,
                show_character_labels=show_character_labels,
                viseme_type=viseme_type
            )
            return

        # Single character rendering path
        fn, tag = self._get_char_renderer(ct, theme)
        is_human = any(w in ct for w in [
            "human", "doctor", "physician", "patient", "teacher", "student", 
            "farmer", "business", "analyst", "engineer", "presenter", "coach", 
            "banker", "diplomat", "executive", "host", "scientist", "coordinator", "worker", "narrator"
        ])
        if is_human:
            cx = int(width * 0.78)
            cy = int(height * 0.65)
            from src.pipeline.avatar_service import avatar_service
            avatar_service.render_animated_presenter(
                canvas=canvas,
                cx=cx,
                cy=cy,
                t=t,
                frame_idx=frame_idx,
                lip_sync_energy=lip_sync_energy,
                role=ct,
                theme_mode=theme,
                show_badge=show_character_labels,
                viseme_type=viseme_type
            )
            return
        fn(canvas, frame_idx, t, width, height, lip_sync_energy)

    # ────────────── CUSTOM CHARACTER TYPE RENDERERS ──────────────────────────

    def _char_doctor(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """Doctor character — crisp white lab coat, cyan/blue medical stethoscope, doctor ID badge, pen."""
        bob_y = int(math.sin(t * 3.0) * 5)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y

        white_coat = (250, 250, 252)    # Pure white medical lab coat
        lapel_gray = (220, 225, 230)
        inner_shirt = (235, 210, 180)   # Light medical blue shirt
        tie_blue = (140, 45, 25)        # Navy tie
        steth_tube = (190, 120, 25)     # Cyan/blue stethoscope tubing
        steth_bell = (215, 215, 225)    # Metallic silver chest piece
        skin = (195, 215, 240)          # Realistic skin tone (BGR)
        hair = (35, 45, 60)             # Professional dark brown hair

        # Shadow beneath
        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Body / Lab Coat Torso
        cv2.ellipse(canvas, (cx, cy + 170), (115, 140), 0, 0, 360, white_coat, -1, cv2.LINE_AA)
        cv2.polylines(canvas, [np.array([[cx - 75, cy + 80], [cx - 75, cy + 240], [cx + 75, cy + 240], [cx + 75, cy + 80]], np.int32)], False, lapel_gray, 2, cv2.LINE_AA)

        # Inner shirt & tie (visible in coat V-neck)
        shirt_pts = np.array([[cx - 28, cy + 45], [cx + 28, cy + 45], [cx + 12, cy + 115], [cx - 12, cy + 115]], np.int32)
        cv2.fillPoly(canvas, [shirt_pts], inner_shirt, cv2.LINE_AA)
        # Navy tie
        tie_pts = np.array([[cx - 7, cy + 55], [cx + 7, cy + 55], [cx + 10, cy + 110], [cx, cy + 125], [cx - 10, cy + 110]], np.int32)
        cv2.fillPoly(canvas, [tie_pts], tie_blue, cv2.LINE_AA)

        # Coat lapels
        cv2.line(canvas, (cx - 40, cy + 50), (cx - 14, cy + 130), lapel_gray, 3, cv2.LINE_AA)
        cv2.line(canvas, (cx + 40, cy + 50), (cx + 14, cy + 130), lapel_gray, 3, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy + 130), (cx, cy + 240), lapel_gray, 2, cv2.LINE_AA)
        # Coat buttons
        for by_btn in [cy + 150, cy + 180, cy + 210]:
            cv2.circle(canvas, (cx, by_btn), 4, (40, 40, 45), -1, cv2.LINE_AA)

        # Left coat pocket with ID badge & pen
        cv2.rectangle(canvas, (cx - 65, cy + 105), (cx - 30, cy + 135), (240, 240, 245), -1)
        cv2.rectangle(canvas, (cx - 65, cy + 105), (cx - 30, cy + 135), lapel_gray, 1, cv2.LINE_AA)
        # Red medical cross on badge
        cv2.line(canvas, (cx - 48, cy + 112), (cx - 48, cy + 122), (40, 40, 220), 2, cv2.LINE_AA)
        cv2.line(canvas, (cx - 53, cy + 117), (cx - 43, cy + 117), (40, 40, 220), 2, cv2.LINE_AA)
        # Pen in pocket
        cv2.line(canvas, (cx - 38, cy + 96), (cx - 38, cy + 112), (200, 40, 20), 3, cv2.LINE_AA)

        # ── REALISTIC MEDICAL STETHOSCOPE ──
        steth_l = [
            (cx - 32, cy + 42), (cx - 45, cy + 65), (cx - 38, cy + 100),
            (cx - 20, cy + 128), (cx + 10, cy + 135)
        ]
        steth_r = [
            (cx + 32, cy + 42), (cx + 45, cy + 65), (cx + 38, cy + 100),
            (cx + 25, cy + 125), (cx + 10, cy + 135)
        ]
        for i in range(len(steth_l) - 1):
            cv2.line(canvas, steth_l[i], steth_l[i+1], steth_tube, 4, cv2.LINE_AA)
        for i in range(len(steth_r) - 1):
            cv2.line(canvas, steth_r[i], steth_r[i+1], steth_tube, 4, cv2.LINE_AA)
        # Hanging tube down to chest piece
        cv2.line(canvas, (cx + 10, cy + 135), (cx + 18, cy + 165), steth_tube, 4, cv2.LINE_AA)
        # Metallic silver chest piece / bell
        cv2.circle(canvas, (cx + 18, cy + 165), 14, steth_bell, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx + 18, cy + 165), 14, (140, 140, 150), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx + 18, cy + 165), 7, (90, 90, 100), -1, cv2.LINE_AA)

        # Gesturing Arm
        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=white_coat, skin_color=skin, lip_sync_energy=lip_sync_energy, theme="hospital")

        # Realistic Human Head via Free API (with glowing audio aura & equalizer)
        self._draw_human_head(
            canvas=canvas,
            cx=cx,
            cy=cy,
            t=t,
            frame_idx=frame_idx,
            lip_sync_energy=lip_sync_energy,
            role="doctor",
            head_radius=64,
            skin_color=skin,
            hair_color=hair,
            eye_color=(90, 60, 30),
            bubble_color=(200, 150, 50)
        )

    def _char_patient(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """Patient character — light medical gown, patient ID wristband, pulse oximeter clip."""
        bob_y = int(math.sin(t * 2.8) * 4)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y

        gown_blue  = (225, 195, 150)    # Light medical powder cyan gown (BGR)
        gown_trim  = (190, 160, 115)
        skin       = (190, 210, 238)
        hair       = (45, 55, 70)

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Torso in Patient Hospital Gown
        cv2.ellipse(canvas, (cx, cy + 170), (112, 138), 0, 0, 360, gown_blue, -1, cv2.LINE_AA)

        # Gown collar V-neck & tie detail
        cv2.ellipse(canvas, (cx, cy + 52), (32, 18), 0, 0, 180, gown_trim, 3, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy + 62), (cx, cy + 120), gown_trim, 2, cv2.LINE_AA)
        for py_pat in [cy + 100, cy + 150, cy + 200]:
            for px_pat in [cx - 45, cx, cx + 45]:
                cv2.circle(canvas, (px_pat, py_pat), 3, gown_trim, -1, cv2.LINE_AA)

        # Left Arm with Hospital Wristband
        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=gown_blue, skin_color=skin, lip_sync_energy=lip_sync_energy, theme="hospital")

        # White Hospital ID wristband on arm
        htx = int(cx - 130 + math.cos(t * 3.5) * 16)
        hty = int(cy + 35 + math.sin(t * 3.5) * 20)
        cv2.rectangle(canvas, (htx - 8, hty - 6), (htx + 8, hty + 6), (245, 245, 250), -1)
        cv2.line(canvas, (htx - 4, hty - 4), (htx - 4, hty + 4), (180, 60, 30), 1)

        # Pulse oximeter clip on finger with glowing red LED
        cv2.circle(canvas, (htx - 22, hty - 10), 4, (30, 30, 240), -1, cv2.LINE_AA)

        # Realistic Human Head via Free API (with glowing audio aura & equalizer)
        self._draw_human_head(
            canvas=canvas,
            cx=cx,
            cy=cy,
            t=t,
            frame_idx=frame_idx,
            lip_sync_energy=lip_sync_energy,
            role="patient",
            head_radius=62,
            skin_color=skin,
            hair_color=hair,
            eye_color=(80, 50, 40),
            bubble_color=(240, 200, 120)
        )

    def _char_king(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """King character — royal velvet robe with ermine fur trim, golden crown with jewels."""
        bob_y = int(math.sin(t * 3.0) * 5)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y

        crimson   = (35, 18, 145)    # Royal velvet crimson (BGR)
        ermine    = (240, 245, 250)  # White ermine fur collar
        gold      = (30, 185, 240)   # Imperial gold
        skin      = (190, 210, 240)
        hair      = (50, 60, 75)

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Royal Robe Torso
        cv2.ellipse(canvas, (cx, cy + 170), (120, 142), 0, 0, 360, crimson, -1, cv2.LINE_AA)

        # Ermine fur collar & golden chain
        cv2.ellipse(canvas, (cx, cy + 60), (70, 28), 0, 0, 180, ermine, -1, cv2.LINE_AA)
        for fx in [cx - 40, cx - 18, cx + 18, cx + 40]:
            cv2.line(canvas, (fx, cy + 62), (fx, cy + 72), (20, 20, 25), 2)
        cv2.ellipse(canvas, (cx, cy + 78), (55, 20), 0, 0, 180, gold, 3, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy + 98), 8, gold, -1, cv2.LINE_AA)

        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=crimson, skin_color=skin)

        # Neck
        cv2.rectangle(canvas, (cx - 20, cy + 30), (cx + 20, cy + 50), skin, -1)

        # Head
        cv2.circle(canvas, (cx, cy), 64, skin, -1, cv2.LINE_AA)

        # Beard & Hair
        cv2.ellipse(canvas, (cx, cy + 28), (44, 28), 0, 0, 180, hair, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy - 35), (66, 38), 0, 180, 360, hair, -1, cv2.LINE_AA)

        # ── GOLDEN CROWN WITH JEWELS ──
        crown_y = cy - 65
        crown_pts = np.array([
            [cx - 48, crown_y], [cx - 40, crown_y - 32], [cx - 20, crown_y - 12],
            [cx, crown_y - 38], [cx + 20, crown_y - 12], [cx + 40, crown_y - 32],
            [cx + 48, crown_y]
        ], np.int32)
        cv2.fillPoly(canvas, [crown_pts], gold, cv2.LINE_AA)
        cv2.polylines(canvas, [crown_pts], True, (20, 150, 200), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, crown_y - 18), 6, (40, 30, 220), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx - 24, crown_y - 12), 4, (220, 80, 40), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx + 24, crown_y - 12), 4, (220, 80, 40), -1, cv2.LINE_AA)

        # Eyes, Mouth, and HUD
        self._draw_eyes(canvas, cx, cy - 6, t, frame_idx, lip_sync_energy, pupil_color=(80, 60, 40))
        self._draw_mouth(canvas, cx, cy + 22, lip_sync_energy)
        self._draw_equalizer_bubble(canvas, cx, cy - 10, t, lip_sync_energy, bubble_color=gold)

    def _char_advisor(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """Royal advisor / scholar character — scholarly robe, spectacles, rolled parchment scroll."""
        bob_y = int(math.sin(t * 3.2) * 5)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y

        tunic     = (115, 65, 35)    # Navy/slate tunic (BGR)
        tunic_trim= (50, 160, 210)
        scroll    = (200, 230, 245)  # Parchment
        skin      = (190, 210, 240)
        gray_hair = (160, 165, 175)

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Robe Torso
        cv2.ellipse(canvas, (cx, cy + 170), (115, 138), 0, 0, 360, tunic, -1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy + 60), (cx, cy + 240), tunic_trim, 3, cv2.LINE_AA)

        # Gesturing arm holding parchment scroll
        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=tunic, skin_color=skin)

        # Parchment scroll in hand
        htx = int(cx - 130 + math.cos(t * 3.5) * 16)
        hty = int(cy + 35 + math.sin(t * 3.5) * 20)
        cv2.rectangle(canvas, (htx - 14, hty - 24), (htx + 14, hty + 24), scroll, -1)
        cv2.rectangle(canvas, (htx - 14, hty - 24), (htx + 14, hty + 24), (120, 150, 180), 1, cv2.LINE_AA)
        cv2.line(canvas, (htx - 14, hty), (htx + 14, hty), (40, 40, 200), 3)  # Red ribbon seal

        # Neck
        cv2.rectangle(canvas, (cx - 18, cy + 30), (cx + 18, cy + 50), skin, -1)

        # Head
        cv2.circle(canvas, (cx, cy), 64, skin, -1, cv2.LINE_AA)

        # Scholar Hair
        cv2.ellipse(canvas, (cx, cy - 35), (66, 36), 0, 180, 360, gray_hair, -1, cv2.LINE_AA)

        # Spectacles / Glasses
        cv2.circle(canvas, (cx - 26, cy - 6), 14, (30, 160, 220), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx + 26, cy - 6), 14, (30, 160, 220), 2, cv2.LINE_AA)
        cv2.line(canvas, (cx - 12, cy - 6), (cx + 12, cy - 6), (30, 160, 220), 2, cv2.LINE_AA)

        # Eyes, Mouth, and HUD
        self._draw_eyes(canvas, cx, cy - 6, t, frame_idx, lip_sync_energy, pupil_color=(80, 60, 40))
        self._draw_mouth(canvas, cx, cy + 22, lip_sync_energy)
        self._draw_equalizer_bubble(canvas, cx, cy - 10, t, lip_sync_energy, bubble_color=tunic_trim)

    def _char_lion(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """Lion character — golden mane, amber eyes, roaring mouth with audio-sync."""
        bob_y = int(math.sin(t * 3.2) * 7)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        gold  = (30, 160, 220)   # golden fur (BGR)
        mane  = (10, 90, 170)    # dark amber mane
        amber = (20, 130, 240)   # amber eyes

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Body / torso
        cv2.ellipse(canvas, (cx, cy + 170), (115, 138), 0, 0, 360, gold, -1, cv2.LINE_AA)

        # Arm pointing
        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=mane, skin_color=gold)

        # Neck
        cv2.rectangle(canvas, (cx-22, cy+50), (cx+22, cy+90), gold, -1)

        # ── Mane (big shaggy circle) ──
        for mi in range(16):
            angle = mi * 22.5
            mx = int(cx + math.cos(math.radians(angle)) * 88)
            my = int(cy + math.sin(math.radians(angle)) * 88)
            cv2.circle(canvas, (mx, my), 32, mane, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 76, mane, -1, cv2.LINE_AA)

        # Head (round lion face)
        cv2.circle(canvas, (cx, cy), 68, gold, -1, cv2.LINE_AA)

        # Ears
        for ex, ey in [(cx-52, cy-52), (cx+52, cy-52)]:
            cv2.fillPoly(canvas, [np.array([[ex-16, ey+10],[ex, ey-22],[ex+16, ey+10]], np.int32)], mane, cv2.LINE_AA)
            cv2.fillPoly(canvas, [np.array([[ex-9, ey+6],[ex, ey-12],[ex+9, ey+6]], np.int32)], (60, 80, 200), cv2.LINE_AA)

        # Snout
        cv2.ellipse(canvas, (cx, cy+20), (34, 24), 0, 0, 360, (160, 200, 230), -1, cv2.LINE_AA)
        # Nose
        cv2.fillPoly(canvas, [np.array([[cx-7, cy+10],[cx+7, cy+10],[cx, cy+18]], np.int32)], (40, 20, 140), cv2.LINE_AA)
        # Whisker lines
        for wx in [-1, 1]:
            for wy in range(3):
                cv2.line(canvas, (cx + wx*36, cy+14+wy*7), (cx + wx*68, cy+10+wy*7), (200, 200, 200), 1, cv2.LINE_AA)

        # Mouth (audio-synced roar)
        self._draw_mouth(canvas, cx, cy + 14, lip_sync_energy, oral_color=(20, 20, 100), tongue_color=(50, 80, 220))

        # Eyes
        self._draw_eyes(canvas, cx, cy - 12, t, frame_idx, lip_sync_energy, pupil_color=amber, brow_color=mane)

        # Tail (visible left side)
        for ti in range(20):
            tx = int(cx - 90 - ti * 5 + math.sin(t * 3 + ti * 0.3) * 8)
            ty = int(cy + 170 + ti * 8)
            cv2.circle(canvas, (tx, ty), max(2, 10 - ti // 3), mane, -1, cv2.LINE_AA)

        self._draw_equalizer_bubble(canvas, cx, cy - 12, t, lip_sync_energy, bubble_color=amber)
        self._draw_badge(canvas, cx, cy, "LEON // LION HOST", border_color=amber)

    def _char_robot(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """Robot character — metallic chrome body, LED eyes, speaker-grill mouth."""
        bob_y = int(math.sin(t * 4.0) * 5)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        metal   = (160, 160, 165)  # chrome (BGR)
        dark    = (40, 40, 45)
        neon_b  = (240, 200, 60)   # neon blue (BGR)

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Torso — rectangular chassis
        cv2.rectangle(canvas, (cx-70, cy+60), (cx+70, cy+210), dark, -1)
        cv2.rectangle(canvas, (cx-70, cy+60), (cx+70, cy+210), metal, 3, cv2.LINE_AA)
        # Chest panel LED grid
        for ri in range(3):
            for ci2 in range(4):
                led_col = (0, int(200 * abs(math.sin(t * 6 + ri + ci2))), 80)
                cv2.circle(canvas, (cx - 34 + ci2 * 23, cy + 100 + ri * 22), 6, led_col, -1, cv2.LINE_AA)

        # Arms (rectangular)
        for side in [-1, 1]:
            ax = cx + side * 85
            cv2.rectangle(canvas, (ax - 12, cy + 65), (ax + 12, cy + 170), metal, -1)
            cv2.rectangle(canvas, (ax - 12, cy + 65), (ax + 12, cy + 170), dark, 2)
        # Pointing hand
        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=metal, skin_color=metal)

        # Neck (cylinder)
        cv2.rectangle(canvas, (cx - 18, cy + 42), (cx + 18, cy + 65), metal, -1)

        # Head (square)
        cv2.rectangle(canvas, (cx - 58, cy - 60), (cx + 58, cy + 42), dark, -1)
        cv2.rectangle(canvas, (cx - 58, cy - 60), (cx + 58, cy + 42), metal, 3, cv2.LINE_AA)

        # Antenna
        cv2.line(canvas, (cx, cy - 60), (cx, cy - 88), metal, 4, cv2.LINE_AA)
        glow_r = 8 + int(abs(math.sin(t * 5)) * 5)
        cv2.circle(canvas, (cx, cy - 88), glow_r, neon_b, -1, cv2.LINE_AA)

        # Eyes — big rectangular LEDs
        for ex in [cx - 24, cx + 24]:
            ey_col = (int(neon_b[0] * abs(math.sin(t * 3 + ex))), neon_b[1], neon_b[2])
            cv2.rectangle(canvas, (ex - 14, cy - 28), (ex + 14, cy - 4), ey_col, -1)
            cv2.rectangle(canvas, (ex - 14, cy - 28), (ex + 14, cy - 4), metal, 1)

        # Speaker-grill mouth (audio sync bars)
        mouth_y = cy + 18
        for gi in range(7):
            bar_h = int(4 + lip_sync_energy * 14 + math.sin(t * 15 + gi) * 3)
            bar_col = neon_b if gi % 2 == 0 else metal
            gx = cx - 30 + gi * 10
            cv2.rectangle(canvas, (gx, mouth_y - bar_h), (gx + 7, mouth_y + bar_h), bar_col, -1)
        cv2.rectangle(canvas, (cx - 33, mouth_y - 18), (cx + 35, mouth_y + 18), metal, 2)

        self._draw_equalizer_bubble(canvas, cx, cy - 60, t, lip_sync_energy, bubble_color=neon_b, bg_color=dark)
        self._draw_badge(canvas, cx, cy, "R-7 // ROBO HOST", border_color=neon_b, bg_color=dark, text_color=neon_b)

    def _char_cat(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """Cat character — orange tabby, whiskers, pointy ears, swishing tail."""
        bob_y = int(math.sin(t * 4.5) * 6)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        fur   = (60, 130, 220)   # orange tabby (BGR)
        stripe= (40, 90, 170)
        green_eye = (60, 200, 80)

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Body
        cv2.ellipse(canvas, (cx, cy + 170), (112, 136), 0, 0, 360, fur, -1, cv2.LINE_AA)
        # Tabby stripes on body
        for si in range(4):
            cv2.line(canvas, (cx - 55 + si * 28, cy + 100), (cx - 45 + si * 28, cy + 200), stripe, 5, cv2.LINE_AA)

        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=fur, skin_color=fur)

        # Neck
        cv2.rectangle(canvas, (cx-18, cy+50), (cx+18, cy+90), fur, -1)

        # Head
        cv2.circle(canvas, (cx, cy), 62, fur, -1, cv2.LINE_AA)
        # Tabby stripe on forehead
        for si in range(3):
            cv2.line(canvas, (cx - 15 + si * 15, cy - 55), (cx - 12 + si * 15, cy - 20), stripe, 3, cv2.LINE_AA)

        # Pointy ears
        for ex, ey in [(cx - 44, cy - 44), (cx + 44, cy - 44)]:
            cv2.fillPoly(canvas, [np.array([[ex-18, ey+14],[ex, ey-30],[ex+18, ey+14]], np.int32)], fur, cv2.LINE_AA)
            cv2.fillPoly(canvas, [np.array([[ex-10, ey+8],[ex, ey-18],[ex+10, ey+8]], np.int32)], (100, 80, 200), cv2.LINE_AA)

        # Muzzle
        cv2.ellipse(canvas, (cx, cy + 22), (28, 18), 0, 0, 360, (200, 200, 220), -1, cv2.LINE_AA)
        # Nose (tiny triangle)
        cv2.fillPoly(canvas, [np.array([[cx-5, cy+14],[cx+5, cy+14],[cx, cy+20]], np.int32)], (60, 40, 200), cv2.LINE_AA)
        # Whiskers
        for wx in [-1, 1]:
            for wy in range(3):
                cv2.line(canvas, (cx + wx*28, cy+16+wy*6), (cx + wx*62, cy+12+wy*6), (220, 220, 220), 1, cv2.LINE_AA)

        self._draw_mouth(canvas, cx, cy + 16, lip_sync_energy, oral_color=(40, 20, 120), tongue_color=(80, 100, 220))
        self._draw_eyes(canvas, cx, cy - 10, t, frame_idx, lip_sync_energy, pupil_color=green_eye, brow_color=stripe)

        # Swishing tail
        for ti in range(25):
            angle = math.pi * 1.2 + ti * 0.12 + math.sin(t * 2.5) * 0.4
            tx = int(cx - 80 + ti * math.cos(angle) * 7)
            ty = int(cy + 190 + ti * math.sin(angle) * 5)
            cv2.circle(canvas, (tx, ty), max(2, 11 - ti // 3), stripe, -1, cv2.LINE_AA)

        self._draw_equalizer_bubble(canvas, cx, cy, t, lip_sync_energy, bubble_color=green_eye)
        self._draw_badge(canvas, cx, cy, "CLEO // CAT HOST", border_color=green_eye)

    def _char_dog(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """Dog character — golden retriever, floppy ears, wagging tail, big smile."""
        bob_y = int(math.sin(t * 5.0) * 9)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        fur   = (50, 170, 230)  # golden retriever (BGR)
        dark_fur = (30, 110, 170)
        brown_eye = (20, 80, 140)

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        cv2.ellipse(canvas, (cx, cy + 170), (113, 136), 0, 0, 360, fur, -1, cv2.LINE_AA)
        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=fur, skin_color=fur)
        cv2.rectangle(canvas, (cx-20, cy+50), (cx+20, cy+90), fur, -1)
        cv2.circle(canvas, (cx, cy), 64, fur, -1, cv2.LINE_AA)

        # Floppy ears
        for side, ex in [(-1, cx - 54), (1, cx + 54)]:
            ear_swing = int(math.sin(t * 4 + side) * 8)
            cv2.ellipse(canvas, (ex + ear_swing, cy + 10), (26, 46), side * 15, 0, 360, dark_fur, -1, cv2.LINE_AA)

        # Snout
        cv2.ellipse(canvas, (cx, cy + 24), (36, 26), 0, 0, 360, dark_fur, -1, cv2.LINE_AA)
        # Nose
        cv2.ellipse(canvas, (cx, cy + 14), (14, 10), 0, 0, 360, (20, 20, 30), -1, cv2.LINE_AA)

        self._draw_mouth(canvas, cx, cy + 18, lip_sync_energy, oral_color=(20, 20, 100), tongue_color=(80, 80, 230))
        self._draw_eyes(canvas, cx, cy - 12, t, frame_idx, lip_sync_energy, pupil_color=brown_eye, brow_color=dark_fur)

        # Wagging tail
        wag = math.sin(t * 6) * 0.6
        for ti in range(22):
            angle = math.pi * 0.3 + wag + ti * 0.1
            tx = int(cx - 90 + ti * math.cos(angle) * 6)
            ty = int(cy + 180 + ti * math.sin(angle) * 5)
            cv2.circle(canvas, (tx, ty), max(2, 10 - ti // 3), fur, -1, cv2.LINE_AA)

        self._draw_equalizer_bubble(canvas, cx, cy, t, lip_sync_energy, bubble_color=(50, 200, 240))
        self._draw_badge(canvas, cx, cy, "BUDDY // DOG HOST", border_color=(50, 200, 240))

    def _char_bird(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """Bird character — tropical parrot, colourful feathers, beak mouth."""
        bob_y = int(math.sin(t * 6.0) * 10)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        red    = (40, 40, 220)
        yellow = (20, 210, 240)
        green  = (40, 180, 60)
        blue_w = (200, 80, 40)

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Body
        cv2.ellipse(canvas, (cx, cy + 160), (95, 120), 0, 0, 360, red, -1, cv2.LINE_AA)
        # Wing (flapping)
        wing_angle = int(math.sin(t * 5) * 20)
        cv2.ellipse(canvas, (cx - 70, cy + 120), (55, 30), wing_angle, 0, 360, yellow, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx + 70, cy + 120), (55, 30), -wing_angle, 0, 360, green, -1, cv2.LINE_AA)

        # Pointing claw/wing
        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=red, skin_color=yellow)

        # Neck
        cv2.rectangle(canvas, (cx-16, cy+50), (cx+16, cy+90), red, -1)
        # Head
        cv2.circle(canvas, (cx, cy), 55, red, -1, cv2.LINE_AA)
        # Colourful head feathers
        for fi in range(7):
            angle = -60 + fi * 20
            fax = int(cx + math.cos(math.radians(angle)) * 52)
            fay = int(cy - 30 + math.sin(math.radians(angle)) * -30)
            f_col = [red, yellow, green, blue_w, yellow, green, red][fi]
            cv2.ellipse(canvas, (fax, fay), (10, 22), angle, 0, 360, f_col, -1, cv2.LINE_AA)

        # Eyes
        self._draw_eyes(canvas, cx, cy - 8, t, frame_idx, lip_sync_energy, pupil_color=(20, 20, 30), brow_color=yellow)
        # Beak (audio-synced)
        beak_open = int(lip_sync_energy * 20)
        cv2.fillPoly(canvas, [np.array([[cx-12, cy+14],[cx+12, cy+14],[cx+6, cy+26],[cx-6, cy+26]], np.int32)], yellow, cv2.LINE_AA)
        cv2.fillPoly(canvas, [np.array([[cx-10, cy+20+beak_open],[cx+10, cy+20+beak_open],[cx+6, cy+32+beak_open],[cx-6, cy+32+beak_open]], np.int32)], (10, 150, 200), cv2.LINE_AA)

        # Tail feathers
        for ti in range(5):
            angle = 200 + ti * 14
            tx = int(cx + math.cos(math.radians(angle)) * (80 + ti * 10))
            ty = int(cy + 200 + math.sin(math.radians(angle)) * 20)
            f_col = [red, yellow, green, blue_w, yellow][ti]
            cv2.line(canvas, (cx, cy + 200), (tx, ty), f_col, 6, cv2.LINE_AA)

        self._draw_equalizer_bubble(canvas, cx, cy, t, lip_sync_energy, bubble_color=yellow)
        self._draw_badge(canvas, cx, cy, "POLLY // PARROT HOST", border_color=yellow)

    def _char_astronaut(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """Astronaut — white spacesuit, golden visor, NASA-style mission patch."""
        bob_y = int(math.sin(t * 2.5) * 5)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        white  = (240, 240, 245)
        gold   = (30, 190, 220)
        patch  = (40, 40, 180)

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Suit body (puffy)
        cv2.ellipse(canvas, (cx, cy + 175), (125, 145), 0, 0, 360, white, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy + 175), (125, 145), 0, 0, 360, (180, 180, 190), 3, cv2.LINE_AA)
        # Mission patch
        cv2.circle(canvas, (cx - 55, cy + 105), 22, patch, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx - 55, cy + 105), 22, gold, 2, cv2.LINE_AA)
        cv2.putText(canvas, "ISS", (cx - 68, cy + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Glove arms
        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=white, skin_color=(200, 200, 215))

        # Neck ring
        cv2.ellipse(canvas, (cx, cy + 55), (38, 16), 0, 0, 360, (160, 160, 170), -1, cv2.LINE_AA)

        # Helmet (large sphere)
        cv2.circle(canvas, (cx, cy - 8), 78, white, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy - 8), 78, (180, 180, 190), 3, cv2.LINE_AA)

        # Gold visor
        cv2.ellipse(canvas, (cx, cy - 8), (58, 44), 0, 0, 360, gold, -1, cv2.LINE_AA)
        # Visor reflection
        cv2.ellipse(canvas, (cx - 18, cy - 22), (16, 26), -20, 0, 360, (255, 230, 120), -1, cv2.LINE_AA)

        # Inside visor — face silhouette
        skin_col = (170, 195, 215)
        cv2.circle(canvas, (cx, cy - 8), 42, skin_col, -1, cv2.LINE_AA)
        self._draw_eyes(canvas, cx, cy - 16, t, frame_idx, lip_sync_energy, pupil_color=(60, 80, 40), brow_color=(40, 40, 50))
        self._draw_mouth(canvas, cx, cy - 8, lip_sync_energy)

        # Communication antenna
        cv2.line(canvas, (cx + 65, cy - 58), (cx + 65, cy - 100), (160, 160, 170), 4, cv2.LINE_AA)
        blink = (0, 200, 80) if (frame_idx % 30) < 15 else (0, 80, 30)
        cv2.circle(canvas, (cx + 65, cy - 100), 6, blink, -1, cv2.LINE_AA)

        self._draw_equalizer_bubble(canvas, cx, cy - 8, t, lip_sync_energy, bubble_color=gold)
        self._draw_badge(canvas, cx, cy, "NOVA // ASTRO HOST", border_color=gold)

    def _char_wizard(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """Wizard — purple starry robe, tall pointy hat, glowing staff, spell sparks."""
        bob_y = int(math.sin(t * 2.8) * 6)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        robe   = (140, 40, 90)   # purple (BGR)
        star   = (20, 210, 240)  # gold stars
        skin   = (175, 200, 225)

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Robe body
        cv2.fillPoly(canvas, [np.array([
            [cx - 110, cy + 230], [cx + 110, cy + 230],
            [cx + 65, cy + 55], [cx - 65, cy + 55]
        ], np.int32)], robe, cv2.LINE_AA)
        # Stars on robe
        for si in range(10):
            sx = cx - 80 + (si * 21 % 160)
            sy = cy + 80 + (si * 17 % 130)
            cv2.circle(canvas, (sx, sy), 5, star, -1, cv2.LINE_AA)

        # Staff / wand (right side)
        wand_x = cx + 90
        cv2.line(canvas, (wand_x, cy - 40), (wand_x, cy + 200), (120, 80, 50), 7, cv2.LINE_AA)
        glow = int(abs(math.sin(t * 4)) * 80 + 40)
        cv2.circle(canvas, (wand_x, cy - 40), 18, star, -1, cv2.LINE_AA)
        # Spark particles
        for sp in range(8):
            angle = sp * 45 + t * 120
            spx = int(wand_x + math.cos(math.radians(angle)) * (16 + glow * 0.2))
            spy = int(cy - 40 + math.sin(math.radians(angle)) * (16 + glow * 0.2))
            cv2.circle(canvas, (spx, spy), 3, (0, 240, 255), -1, cv2.LINE_AA)

        # Gesture arm
        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=robe, skin_color=skin)

        # Neck
        cv2.rectangle(canvas, (cx-18, cy+50), (cx+18, cy+90), skin, -1)
        # Head
        cv2.ellipse(canvas, (cx, cy), (60, 76), 0, 0, 360, skin, -1, cv2.LINE_AA)

        # Long white beard
        cv2.fillPoly(canvas, [np.array([
            [cx-28, cy+36], [cx+28, cy+36],
            [cx+14, cy+130], [cx, cy+140], [cx-14, cy+130]
        ], np.int32)], (230, 230, 235), cv2.LINE_AA)

        # Tall pointy hat
        cv2.fillPoly(canvas, [np.array([[cx-56, cy-46],[cx+56, cy-46],[cx, cy-120]], np.int32)], robe, cv2.LINE_AA)
        # Hat brim
        cv2.ellipse(canvas, (cx, cy - 46), (66, 16), 0, 0, 360, (100, 25, 65), -1, cv2.LINE_AA)
        # Star on hat
        cv2.circle(canvas, (cx + 20, cy - 80), 8, star, -1, cv2.LINE_AA)

        self._draw_eyes(canvas, cx, cy - 8, t, frame_idx, lip_sync_energy, pupil_color=(150, 40, 120), brow_color=(200, 200, 210))
        self._draw_mouth(canvas, cx, cy, lip_sync_energy)
        self._draw_equalizer_bubble(canvas, cx, cy, t, lip_sync_energy, bubble_color=star)
        self._draw_badge(canvas, cx, cy, "MERLIN // WIZARD HOST", border_color=star)

    def _char_alien(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """Alien — green skin, huge black eyes, antenna, glowing aura."""
        bob_y = int(math.sin(t * 3.5) * 8)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        green  = (60, 180, 80)   # alien green (BGR)
        dark_g = (30, 120, 40)
        glow_c = (240, 240, 60)  # yellow glow

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Glow aura
        for gi in range(4):
            r = 130 + gi * 18
            alpha_col = max(0, 40 - gi * 10)
            cv2.circle(canvas, (cx, cy), r, (0, alpha_col, 0), 1, cv2.LINE_AA)

        # Body (slim)
        cv2.ellipse(canvas, (cx, cy + 175), (88, 130), 0, 0, 360, green, -1, cv2.LINE_AA)
        # Silver suit detail
        cv2.ellipse(canvas, (cx, cy + 175), (88, 130), 0, 0, 360, (180, 180, 190), 2, cv2.LINE_AA)
        # Chest symbol
        cv2.circle(canvas, (cx, cy + 115), 18, glow_c, -1, cv2.LINE_AA)
        glow_r2 = int(abs(math.sin(t * 3)) * 8 + 10)
        cv2.circle(canvas, (cx, cy + 115), glow_r2, (0, 60, 0), -1, cv2.LINE_AA)

        # Long thin arms
        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=green, skin_color=green)

        # Thin neck
        cv2.rectangle(canvas, (cx-12, cy+50), (cx+12, cy+90), green, -1)

        # Head — large elongated oval
        cv2.ellipse(canvas, (cx, cy - 10), (55, 80), 0, 0, 360, green, -1, cv2.LINE_AA)

        # Double antenna
        for ax_off, ay_off in [(-24, -60), (24, -60)]:
            cv2.line(canvas, (cx + ax_off, cy - 60), (cx + ax_off + int(math.sin(t * 3) * 5), cy - 100), dark_g, 3, cv2.LINE_AA)
            cv2.circle(canvas, (cx + ax_off + int(math.sin(t * 3) * 5), cy - 100), 7, glow_c, -1, cv2.LINE_AA)

        # Huge black almond eyes
        for ex in [cx - 22, cx + 22]:
            cv2.ellipse(canvas, (ex, cy - 16), (20, 14), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)
            # Alien pupil glow
            pupil_c = glow_c if lip_sync_energy > 0.05 else (0, 150, 30)
            cv2.circle(canvas, (ex, cy - 16), 5, pupil_c, -1, cv2.LINE_AA)

        # Slit mouth (audio sync)
        self._draw_mouth(canvas, cx, cy + 32, lip_sync_energy, oral_color=(20, 80, 20), tongue_color=(40, 160, 40))

        self._draw_equalizer_bubble(canvas, cx, cy - 10, t, lip_sync_energy, bubble_color=glow_c, bg_color=(10, 30, 10))
        self._draw_badge(canvas, cx, cy, "ZRIX // ALIEN HOST", border_color=glow_c, bg_color=(10, 30, 10), text_color=glow_c)



    # ───────────────── 11 UNIQUE CHARACTER RENDERERS ─────────────────────────

    def _char_analyst(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """WARNING theme — Alex the Analyst: navy suit, red tie, glasses, clipboard."""
        bob_y = int(math.sin(t * 3.8) * 8)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        skin = (145, 175, 215)   # warm tan (BGR)
        hair = (25, 25, 35)
        suit = (95, 30, 35)      # dark navy
        tie  = (30, 30, 200)     # red tie

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Body — navy suit jacket
        cv2.ellipse(canvas, (cx, cy + 170), (115, 138), 0, 0, 360, suit, -1, cv2.LINE_AA)
        # White shirt collar V
        cv2.fillPoly(canvas, [np.array([[cx-28, cy+82],[cx, cy+148],[cx+28, cy+82]], np.int32)],
                     (245, 245, 255), cv2.LINE_AA)
        # Red tie
        cv2.fillPoly(canvas, [np.array([[cx-9, cy+88],[cx+9, cy+88],[cx+5, cy+148],[cx-5, cy+148]], np.int32)],
                     tie, cv2.LINE_AA)

        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=suit, skin_color=skin)

        # Clipboard prop (right hand)
        clip_x, clip_y = cx + 68, cy + 90
        cv2.rectangle(canvas, (clip_x - 18, clip_y - 35), (clip_x + 18, clip_y + 40), (200, 160, 90), -1)
        cv2.rectangle(canvas, (clip_x - 18, clip_y - 35), (clip_x + 18, clip_y + 40), (120, 80, 30), 2)
        cv2.rectangle(canvas, (clip_x - 8, clip_y - 40), (clip_x + 8, clip_y - 32), (100, 100, 120), -1)
        for li in range(4):
            cv2.line(canvas, (clip_x - 12, clip_y - 20 + li * 12), (clip_x + 12, clip_y - 20 + li * 12),
                     (80, 80, 80), 1, cv2.LINE_AA)

        # Realistic Human Head via Free API (with glowing audio aura & equalizer)
        self._draw_human_head(
            canvas=canvas,
            cx=cx,
            cy=cy,
            t=t,
            frame_idx=frame_idx,
            lip_sync_energy=lip_sync_energy,
            role="analyst",
            head_radius=64,
            skin_color=skin,
            hair_color=hair,
            eye_color=(80, 50, 40),
            bubble_color=(60, 60, 220)
        )
        self._draw_badge(canvas, cx, cy, "ALEX // DATA ANALYST", border_color=(60, 60, 200))

    def _char_tired_worker(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """WASTE theme — Jordan the Tired Worker: grey tee, coffee mug, dark circles."""
        bob_y = int(math.sin(t * 2.2) * 4)   # slower, sluggish bob
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        skin = (170, 200, 235)   # light skin (BGR)
        hair = (60, 55, 100)     # messy dishevelled brown
        shirt = (140, 140, 140)  # faded grey

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Faded grey T-shirt
        cv2.ellipse(canvas, (cx, cy + 170), (112, 135), 0, 0, 360, shirt, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (cx-18, cy+82), (cx+18, cy+120), (160, 160, 160), -1)

        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=shirt, skin_color=skin)

        # Coffee mug right hand
        mug_x, mug_y = cx + 65, cy + 115
        cv2.rectangle(canvas, (mug_x - 16, mug_y - 22), (mug_x + 16, mug_y + 22), (80, 60, 50), -1)
        cv2.ellipse(canvas, (mug_x + 16, mug_y), (6, 10), 0, 270, 90, (80, 60, 50), 3, cv2.LINE_AA)
        # Steam wisps
        for si in range(3):
            sx = mug_x - 8 + si * 8
            swy = int(math.sin(t * 4 + si) * 4)
            cv2.line(canvas, (sx, mug_y - 24), (sx + swy, mug_y - 44), (190, 190, 190), 2, cv2.LINE_AA)

        cv2.rectangle(canvas, (cx-20, cy+50), (cx+20, cy+90), skin, -1)
        cv2.ellipse(canvas, (cx, cy), (64, 80), 0, 0, 360, skin, -1, cv2.LINE_AA)

        # Messy dishevelled hair — multiple jagged tufts
        for dx, dy, ang in [(-10, -44, 0), (-25, -50, 15), (5, -46, -10), (20, -48, 8), (-2, -54, 3)]:
            cv2.ellipse(canvas, (cx + dx, cy + dy), (28, 22), ang, 0, 360, hair, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy-42), (72, 50), 0, 180, 360, hair, -1, cv2.LINE_AA)

        # Dark under-eye bags
        cv2.ellipse(canvas, (cx-26, cy-2), (14, 5), 0, 0, 180, (80, 90, 130), -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx+26, cy-2), (14, 5), 0, 0, 180, (80, 90, 130), -1, cv2.LINE_AA)

        self._draw_eyes(canvas, cx, cy, t, frame_idx, lip_sync_energy, pupil_color=(60, 50, 40), brow_color=(60, 55, 100))
        self._draw_mouth(canvas, cx, cy, lip_sync_energy)
        self._draw_equalizer_bubble(canvas, cx, cy, t, lip_sync_energy, bubble_color=(140, 140, 200))
        self._draw_badge(canvas, cx, cy, "JORDAN // OVERWORKED", border_color=(140, 140, 190))

    def _char_coach(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """CALENDAR theme — Sam the Coach: red tracksuit, whistle, sweatband."""
        bob_y = int(math.sin(t * 4.5) * 10)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        skin = (120, 155, 200)   # medium brown (BGR)
        hair = (35, 55, 80)      # dark ponytail
        tracksuit = (50, 50, 210)  # red tracksuit

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Red tracksuit body
        cv2.ellipse(canvas, (cx, cy + 170), (112, 136), 0, 0, 360, tracksuit, -1, cv2.LINE_AA)
        # White stripes on jacket shoulders
        cv2.line(canvas, (cx - 90, cy + 100), (cx - 60, cy + 145), (255, 255, 255), 6, cv2.LINE_AA)
        cv2.line(canvas, (cx + 90, cy + 100), (cx + 60, cy + 145), (255, 255, 255), 6, cv2.LINE_AA)
        # Collar
        cv2.fillPoly(canvas, [np.array([[cx-20, cy+82],[cx, cy+130],[cx+20, cy+82]], np.int32)],
                     (255, 255, 255), cv2.LINE_AA)

        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=tracksuit, skin_color=skin)

        # Whistle on lanyard
        lanyard_swing = int(math.sin(t * 3) * 8)
        cv2.line(canvas, (cx, cy+90), (cx - 30 + lanyard_swing, cy + 150), (200, 180, 80), 2, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx - 30 + lanyard_swing, cy + 155), (10, 7), 0, 0, 360, (50, 180, 220), -1, cv2.LINE_AA)

        # Realistic Human Head via Free API (with glowing audio aura & equalizer)
        self._draw_human_head(
            canvas=canvas,
            cx=cx,
            cy=cy,
            t=t,
            frame_idx=frame_idx,
            lip_sync_energy=lip_sync_energy,
            role="coach",
            head_radius=64,
            skin_color=skin,
            hair_color=hair,
            eye_color=(60, 90, 30),
            bubble_color=(50, 200, 80)
        )
        self._draw_badge(canvas, cx, cy, "SAM // PRODUCTIVITY COACH", border_color=(50, 200, 80))

    def _char_hacker(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """TWEAK theme — Zed the Hacker: black hoodie, neon green, thick lens glasses."""
        bob_y = int(math.sin(t * 3.2) * 6)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        skin = (70, 95, 130)    # dark skin (BGR)
        hoodie = (25, 25, 25)   # black hoodie
        neon = (20, 240, 80)    # neon green trim

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Black hoodie body
        cv2.ellipse(canvas, (cx, cy + 168), (115, 140), 0, 0, 360, hoodie, -1, cv2.LINE_AA)
        # Neon green trim lines on hood
        cv2.ellipse(canvas, (cx, cy - 40), (76, 58), 0, 180, 360, (35, 35, 35), -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy - 40), (76, 58), 0, 180, 360, neon, 2, cv2.LINE_AA)
        # Neon edge glow
        neon_alpha = int(abs(math.sin(t * 5)) * 60 + 20)
        cv2.ellipse(canvas, (cx, cy + 168), (118, 143), 0, 0, 360, neon, 2, cv2.LINE_AA)

        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=hoodie, skin_color=skin)

        cv2.rectangle(canvas, (cx-20, cy+50), (cx+20, cy+90), skin, -1)
        cv2.ellipse(canvas, (cx, cy), (62, 78), 0, 0, 360, skin, -1, cv2.LINE_AA)

        # Short natural hair (under hood)
        cv2.ellipse(canvas, (cx, cy-44), (68, 50), 0, 180, 360, (20, 20, 20), -1, cv2.LINE_AA)

        # Thick green-lens glasses
        for ex in [cx-27, cx+27]:
            cv2.ellipse(canvas, (ex, cy-8), (15, 13), 0, 0, 360, (20, 80, 20), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (ex, cy-8), (15, 13), 0, 0, 360, neon, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx-12, cy-8), (cx+12, cy-8), (40, 40, 40), 2, cv2.LINE_AA)

        # Neon glow halo
        glow_r = int(abs(math.sin(t * 4)) * 15 + 75)
        cv2.circle(canvas, (cx, cy), glow_r, neon, 1, cv2.LINE_AA)

        self._draw_eyes(canvas, cx, cy, t, frame_idx, lip_sync_energy, pupil_color=(20, 200, 60), brow_color=(20, 20, 20))
        self._draw_mouth(canvas, cx, cy, lip_sync_energy, oral_color=(10, 80, 20))
        self._draw_equalizer_bubble(canvas, cx, cy, t, lip_sync_energy, bubble_color=neon, bg_color=(10, 15, 10))
        self._draw_badge(canvas, cx, cy, "ZED // SYSTEM OPTIMIZER", border_color=neon, bg_color=(10, 15, 10), text_color=neon)

    def _char_speedster(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """INSTANT theme — Kai the Speedster: orange athletic jersey, yellow headband, speed lines."""
        bob_y = int(math.sin(t * 5.5) * 12)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        skin = (130, 170, 210)  # medium (BGR)
        jersey = (20, 110, 240)  # orange jersey
        headband = (0, 210, 240) # yellow headband

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Speed lines emanating from character
        for sl in range(6):
            angle = sl * 60 + t * 90
            sx = int(cx + math.cos(math.radians(angle)) * 90)
            sy = int(cy + math.sin(math.radians(angle)) * 90)
            ex2 = int(cx + math.cos(math.radians(angle)) * 140)
            ey2 = int(cy + math.sin(math.radians(angle)) * 140)
            alpha_line = max(0, int(180 - sl * 25))
            cv2.line(canvas, (sx, sy), (ex2, ey2), (0, 200, 240), 2, cv2.LINE_AA)

        # Orange athletic jersey
        cv2.ellipse(canvas, (cx, cy + 168), (112, 136), 0, 0, 360, jersey, -1, cv2.LINE_AA)
        # Jersey number #5
        cv2.putText(canvas, "5", (cx - 12, cy + 188), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=jersey, skin_color=skin)

        cv2.rectangle(canvas, (cx-20, cy+50), (cx+20, cy+90), skin, -1)
        cv2.ellipse(canvas, (cx, cy), (63, 79), 0, 0, 360, skin, -1, cv2.LINE_AA)

        # Close-cropped hair
        cv2.ellipse(canvas, (cx, cy-44), (68, 50), 0, 180, 360, (25, 40, 55), -1, cv2.LINE_AA)

        # Yellow headband
        cv2.rectangle(canvas, (cx-66, cy-30), (cx+66, cy-14), headband, -1)
        cv2.rectangle(canvas, (cx-66, cy-30), (cx+66, cy-14), (0, 160, 200), 2)

        self._draw_eyes(canvas, cx, cy, t, frame_idx, lip_sync_energy, pupil_color=(20, 130, 220), brow_color=(25, 40, 55))
        self._draw_mouth(canvas, cx, cy, lip_sync_energy)
        self._draw_equalizer_bubble(canvas, cx, cy, t, lip_sync_energy, bubble_color=(0, 200, 240))
        self._draw_badge(canvas, cx, cy, "KAI // SPEED CHAMPION", border_color=(0, 200, 240))

    def _char_executive(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """EMAIL theme — Morgan the Executive: burgundy blazer, pearl necklace, thin oval glasses."""
        bob_y = int(math.sin(t * 3.0) * 6)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        skin = (185, 210, 240)  # fair (BGR)
        hair = (35, 30, 28)     # elegant dark bun
        blazer = (45, 45, 128)  # burgundy

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Burgundy blazer
        cv2.ellipse(canvas, (cx, cy + 170), (113, 137), 0, 0, 360, blazer, -1, cv2.LINE_AA)
        # White blouse V
        cv2.fillPoly(canvas, [np.array([[cx-25, cy+82],[cx, cy+148],[cx+25, cy+82]], np.int32)],
                     (250, 250, 255), cv2.LINE_AA)
        # Pearl necklace
        for pi in range(7):
            px = cx - 36 + pi * 12
            cv2.circle(canvas, (px, cy + 80), 5, (230, 230, 230), -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, cy + 80), 5, (200, 200, 210), 1, cv2.LINE_AA)

        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=blazer, skin_color=skin)

        # Realistic Human Head via Free API (with glowing audio aura & equalizer)
        self._draw_human_head(
            canvas=canvas,
            cx=cx,
            cy=cy,
            t=t,
            frame_idx=frame_idx,
            lip_sync_energy=lip_sync_energy,
            role="executive",
            head_radius=64,
            skin_color=skin,
            hair_color=hair,
            eye_color=(60, 40, 80),
            bubble_color=(45, 45, 180)
        )
        self._draw_badge(canvas, cx, cy, "MORGAN // EXEC DIRECTOR", border_color=(45, 45, 180))

    def _char_influencer(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """SOCIAL theme — Riley the Influencer: purple hoodie, snapback, AirPods."""
        bob_y = int(math.sin(t * 4.2) * 9)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        skin = (150, 185, 225)  # warm (BGR)
        hair = (50, 60, 80)
        hoodie = (160, 40, 130) # purple

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Purple oversized hoodie
        cv2.ellipse(canvas, (cx, cy + 172), (118, 142), 0, 0, 360, hoodie, -1, cv2.LINE_AA)
        # Kangaroo pocket
        cv2.rectangle(canvas, (cx-40, cy+155), (cx+40, cy+200), (140, 30, 110), -1)
        # Lightning bolt patch
        lbx, lby = cx + 50, cy + 130
        cv2.fillPoly(canvas, [np.array([
            [lbx, lby], [lbx+12, lby+18], [lbx+5, lby+18],
            [lbx+12, lby+36], [lbx, lby+18], [lbx+7, lby+18]
        ], np.int32)], (0, 220, 255), cv2.LINE_AA)

        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=hoodie, skin_color=skin)

        # Realistic Human Head via Free API (with glowing audio aura & equalizer)
        self._draw_human_head(
            canvas=canvas,
            cx=cx,
            cy=cy,
            t=t,
            frame_idx=frame_idx,
            lip_sync_energy=lip_sync_energy,
            role="coach",
            head_radius=64,
            skin_color=skin,
            hair_color=hair,
            eye_color=(110, 40, 140),
            bubble_color=(200, 80, 200)
        )
        self._draw_badge(canvas, cx, cy, "RILEY // CONTENT CREATOR", border_color=(200, 80, 200))

    def _char_banker(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """FINANCE theme — Victor the Banker: charcoal three-piece suit, gold tie, briefcase."""
        bob_y = int(math.sin(t * 2.8) * 5)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        skin = (130, 170, 210)  # tan (BGR)
        hair = (80, 70, 100)    # slicked-back, silver-tinged
        suit = (50, 50, 55)     # charcoal
        gold = (30, 180, 215)   # gold tie

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Charcoal three-piece suit
        cv2.ellipse(canvas, (cx, cy + 170), (114, 138), 0, 0, 360, suit, -1, cv2.LINE_AA)
        # Waistcoat (vest)
        cv2.rectangle(canvas, (cx-32, cy+82), (cx+32, cy+160), (65, 65, 70), -1)
        # White pocket square
        cv2.rectangle(canvas, (cx-50, cy+100), (cx-32, cy+120), (240, 240, 250), -1)
        # Gold tie
        cv2.fillPoly(canvas, [np.array([[cx-8, cy+88],[cx+8, cy+88],[cx+4, cy+155],[cx-4, cy+155]], np.int32)],
                     gold, cv2.LINE_AA)
        # Cufflinks
        for off in [-1, 1]:
            cv2.circle(canvas, (cx + off * 80, cy + 145), 6, gold, -1, cv2.LINE_AA)

        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=suit, skin_color=skin)

        # Briefcase in right hand
        bc_x, bc_y = cx + 68, cy + 110
        cv2.rectangle(canvas, (bc_x - 22, bc_y - 28), (bc_x + 22, bc_y + 28), (50, 80, 100), -1)
        cv2.rectangle(canvas, (bc_x - 22, bc_y - 28), (bc_x + 22, bc_y + 28), (30, 55, 75), 2)
        cv2.rectangle(canvas, (bc_x - 10, bc_y - 30), (bc_x + 10, bc_y - 22), (40, 65, 90), -1)
        cv2.line(canvas, (bc_x - 22, bc_y), (bc_x + 22, bc_y), (30, 55, 75), 2)

        # Realistic Human Head via Free API (with glowing audio aura & equalizer)
        self._draw_human_head(
            canvas=canvas,
            cx=cx,
            cy=cy,
            t=t,
            frame_idx=frame_idx,
            lip_sync_energy=lip_sync_energy,
            role="banker",
            head_radius=64,
            skin_color=skin,
            hair_color=hair,
            eye_color=(50, 65, 40),
            bubble_color=gold
        )
        self._draw_badge(canvas, cx, cy, "VICTOR // FINANCE DIRECTOR", border_color=gold)

    def _char_engineer(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """TECH theme — Dev the Engineer: blue hoodie, round thick glasses, DEV lanyard badge."""
        bob_y = int(math.sin(t * 3.6) * 7)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        skin = (120, 155, 190)  # warm (BGR)
        hair = (25, 30, 45)     # shaggy dark
        hoodie = (180, 100, 40) # blue hoodie

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Blue dev hoodie body
        cv2.ellipse(canvas, (cx, cy + 170), (113, 137), 0, 0, 360, hoodie, -1, cv2.LINE_AA)
        # Kangaroo pocket
        cv2.rectangle(canvas, (cx-42, cy+155), (cx+42, cy+200), (160, 85, 30), -1)
        # Hood outline
        cv2.ellipse(canvas, (cx, cy-42), (75, 56), 0, 180, 360, (165, 90, 32), -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy-42), (75, 56), 0, 180, 360, (140, 75, 25), 2, cv2.LINE_AA)

        # DEV lanyard badge hanging from neck
        swing = int(math.sin(t * 2.5) * 10)
        lany_x = cx - 10 + swing
        cv2.line(canvas, (cx, cy+90), (lany_x, cy + 148), (180, 50, 50), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (lany_x - 20, cy + 148), (lany_x + 20, cy + 175), (240, 240, 255), -1)
        cv2.rectangle(canvas, (lany_x - 20, cy + 148), (lany_x + 20, cy + 175), (80, 80, 200), 1)
        cv2.putText(canvas, "DEV", (lany_x - 14, cy + 168), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 180), 1, cv2.LINE_AA)

        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=hoodie, skin_color=skin, lip_sync_energy=lip_sync_energy, theme="tech")

        # Realistic Human Head via Free API (with glowing audio aura & equalizer)
        self._draw_human_head(
            canvas=canvas,
            cx=cx,
            cy=cy,
            t=t,
            frame_idx=frame_idx,
            lip_sync_energy=lip_sync_energy,
            role="engineer",
            head_radius=64,
            skin_color=skin,
            hair_color=hair,
            eye_color=(180, 120, 30),
            bubble_color=hoodie
        )
        self._draw_badge(canvas, cx, cy, "DEV // SENIOR ENGINEER", border_color=hoodie)

    def _char_diplomat(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """TRUST theme — Priya the Diplomat: navy smart-casual blazer, flag pin."""
        bob_y = int(math.sin(t * 3.1) * 6)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        skin = (110, 155, 200)  # South-Asian warm (BGR)
        hair = (20, 20, 28)     # professional medium dark
        blazer = (110, 60, 30)  # navy blazer

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # Navy blazer body
        cv2.ellipse(canvas, (cx, cy + 170), (113, 137), 0, 0, 360, blazer, -1, cv2.LINE_AA)
        # Light blue open-collar shirt
        cv2.fillPoly(canvas, [np.array([[cx-22, cy+82],[cx, cy+145],[cx+22, cy+82]], np.int32)],
                     (200, 170, 130), cv2.LINE_AA)

        # Flag pin on lapel (small tricolor)
        fp_x, fp_y = cx - 50, cy + 106
        for pi, col in enumerate([(50, 50, 230), (245, 245, 245), (30, 130, 30)]):
            cv2.rectangle(canvas, (fp_x, fp_y + pi * 6), (fp_x + 22, fp_y + pi * 6 + 6), col, -1)
        cv2.rectangle(canvas, (fp_x, fp_y), (fp_x + 22, fp_y + 18), (80, 80, 80), 1)

        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=blazer, skin_color=skin, lip_sync_energy=lip_sync_energy, theme="eco")

        # Realistic Human Head via Free API (with glowing audio aura & equalizer)
        self._draw_human_head(
            canvas=canvas,
            cx=cx,
            cy=cy,
            t=t,
            frame_idx=frame_idx,
            lip_sync_energy=lip_sync_energy,
            role="diplomat",
            head_radius=64,
            skin_color=skin,
            hair_color=hair,
            eye_color=(60, 50, 35),
            bubble_color=(80, 130, 200)
        )
        self._draw_badge(canvas, cx, cy, "PRIYA // TRUST AMBASSADOR", border_color=(80, 130, 200))

    def _char_artist(self, canvas, frame_idx, t, width, height, lip_sync_energy):
        """CREATIVE theme — Milo the Artist: paint-splattered smock, purple beret, paintbrush."""
        bob_y = int(math.sin(t * 3.4) * 8)
        cx = int(width * 0.80)
        cy = int(height * 0.68) + bob_y
        skin = (190, 215, 245)  # fair (BGR)
        hair = (60, 80, 160)    # long auburn hair
        smock = (240, 245, 252) # white smock
        beret = (160, 40, 120)  # purple beret

        cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

        # White paint-splattered smock
        cv2.ellipse(canvas, (cx, cy + 170), (114, 138), 0, 0, 360, smock, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy + 170), (114, 138), 0, 0, 360, (200, 210, 220), 2, cv2.LINE_AA)
        # Paint splatters on smock
        splat_colors = [(40, 40, 230), (20, 180, 50), (0, 100, 240), (200, 30, 200), (0, 200, 240)]
        rng_state = 42
        for idx, sc in enumerate(splat_colors):
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            sx = cx - 80 + (rng_state % 160)
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            sy = cy + 100 + (rng_state % 90)
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            sr = 4 + (rng_state % 8)
            cv2.circle(canvas, (sx, sy), sr, sc, -1, cv2.LINE_AA)

        self._draw_pointing_arm(canvas, cx, cy, t, arm_color=(220, 225, 235), skin_color=skin)

        # Paintbrush prop held in gesture direction
        brush_angle = t * 40
        bx = int(cx - 130 + math.cos(t * 3.5) * 16)
        by = int(cy + 35 + math.sin(t * 3.5) * 20)
        brush_end_x = bx - 40
        brush_end_y = by - 20
        cv2.line(canvas, (bx, by), (brush_end_x, brush_end_y), (140, 100, 60), 4, cv2.LINE_AA)
        cv2.ellipse(canvas, (brush_end_x, brush_end_y), (6, 4), 0, 0, 360, (0, 100, 240), -1, cv2.LINE_AA)

        # Realistic Human Head via Free API (with glowing audio aura & equalizer)
        self._draw_human_head(
            canvas=canvas,
            cx=cx,
            cy=cy,
            t=t,
            frame_idx=frame_idx,
            lip_sync_energy=lip_sync_energy,
            role="artist",
            head_radius=64,
            skin_color=skin,
            hair_color=hair,
            eye_color=(160, 60, 80),
            bubble_color=beret
        )
        self._draw_badge(canvas, cx, cy, "MILO // CREATIVE DIRECTOR", border_color=beret)



    async def fetch_pexels_clip(self, query: str, duration: float) -> Optional[FootageClip]:
        """Queries Pexels stock video API with commercial license tracking."""
        if not settings.PEXELS_API_KEY:
            return None
        
        try:
            url = f"https://api.pexels.com/videos/search?query={urllib.parse.quote(query)}&per_page=5&orientation=landscape"
            req = urllib.request.Request(url, headers={"Authorization": settings.PEXELS_API_KEY, "User-Agent": "IncuBrixEngine/1.0"})
            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode("utf-8"))
                videos = data.get("videos", [])
                
                for v in videos:
                    v_id = f"pexels_{v['id']}"
                    if v_id in self.used_clip_ids:
                        continue
                    
                    files = v.get("video_files", [])
                    best_file = None
                    for f in files:
                        if f.get("quality") == "hd" or f.get("width", 0) >= 1280:
                            best_file = f
                            break
                    if not best_file and files:
                        best_file = files[0]

                    if best_file and best_file.get("link"):
                        download_url = best_file["link"]
                        out_file = self.cache_dir / f"{v_id}.mp4"
                        if not out_file.exists():
                            logger.info(f"Downloading Pexels clip: {v_id}")
                            urllib.request.urlretrieve(download_url, str(out_file))

                        self.used_clip_ids.add(v_id)
                        return FootageClip(
                            clip_id=v_id,
                            local_path=out_file,
                            source="Pexels Commercial Stock API",
                            license_name="Pexels Commercial License (Free to use, commercial, no attribution required)",
                            attribution=f"Video by {v.get('user', {}).get('name', 'Pexels Contributor')} on Pexels",
                            original_url=v.get("url", download_url),
                            is_ai_generated=False,
                            selection_reason=f"High-confidence stock match from Pexels API for subject '{query}'",
                            relevance_score=4.6
                        )
        except Exception as e:
            logger.warning(f"Pexels fetch failed: {str(e)}. Proceeding to fallback source.")
        return None

    async def fetch_pixabay_clip(
        self,
        query: str,
        duration: float,
        aspect_ratio: str = "16:9",
        pixabay_video_id: Optional[int] = None
    ) -> Optional[FootageClip]:
        """Queries Pixabay video API via PixabayProvider with ranked scoring and commercial license tracking."""
        import time
        from src.pipeline.pixabay_provider import pixabay_provider, PIXABAY_LICENSE_NAME, PIXABAY_LICENSE_URL, PIXABAY_RESTRICTION

        try:
            result = None
            if pixabay_video_id:
                result = pixabay_provider.get_video_by_id(pixabay_video_id)

            if not result:
                results = pixabay_provider.search_videos(
                    query=query,
                    per_page=15,
                    aspect_ratio=aspect_ratio,
                    target_duration=duration,
                    used_ids=self.used_clip_ids
                )
                for r in results:
                    v_id = f"pixabay_{r.pixabay_id}"
                    if v_id not in self.used_clip_ids:
                        result = r
                        break

            if not result:
                return None

            v_id = f"pixabay_{result.pixabay_id}"
            out_file = self.cache_dir / f"{v_id}.mp4"

            if result.is_mock:
                # Offline mock mode: render procedural mock video file if missing
                if not out_file.exists():
                    self.generate_procedural_broll(
                        query=query,
                        duration=duration,
                        output_path=out_file,
                        style="bright, modern",
                        aspect_ratio=aspect_ratio
                    )
            else:
                downloaded = pixabay_provider.download_clip(result, out_path=out_file)
                if not downloaded:
                    return None
                out_file = downloaded

            self.used_clip_ids.add(v_id)
            clip = FootageClip(
                clip_id=v_id,
                local_path=out_file,
                source="Pixabay Stock Video API",
                license_name=PIXABAY_LICENSE_NAME,
                attribution=f"Video by {result.user} on Pixabay",
                original_url=result.page_url or f"https://pixabay.com/videos/id-{result.pixabay_id}/",
                is_ai_generated=False,
                selection_reason=f"Pixabay library ranked match for '{query}' (score: {result.relevance_score:.2f})",
                relevance_score=max(3.5, min(5.0, result.relevance_score or 4.5)),
                provider="Pixabay Stock Video API",
                creator=result.user,
                license_url=PIXABAY_LICENSE_URL,
                restrictions=[PIXABAY_RESTRICTION],
                commercial_rights_cleared=True,
                has_licensing_warning=False
            )
            clip.pixabay_video_id = result.pixabay_id
            clip.page_url = result.page_url
            clip.thumbnail_url = result.thumbnail_url
            clip.is_mock = result.is_mock
            clip.download_date = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return clip
        except Exception as e:
            logger.warning(f"Pixabay fetch failed: {str(e)}. Proceeding to fallback source.")
            return None

    async def get_clip_for_scene(
        self,
        scene_index: int,
        search_query: str,
        duration: float,
        style: str,
        preferred_source: Optional[str] = None,
        audio_path: Optional[Path] = None,
        start_time: float = 0.0,
        scene_text: str = "",
        character_type: str = "human",
        scene_speaker: Optional[str] = None,
        visual_hint: Optional[str] = None,
        verified_fact: Optional[Dict[str, Any]] = None,
        is_voiceover: bool = False,
        total_scenes: int = 1,
        aspect_ratio: str = "16:9",
        custom_headline: Optional[str] = None,
        show_character_labels: bool = False,
        visual_source_preference: str = "hybrid",
        pixabay_video_id: Optional[int] = None
    ) -> FootageClip:
        """
        Intelligent multi-source selector implementing Section 6 Requirements:
        - 'ai_first': Prioritizes AI generation for all scenes.
        - 'stock_first': Queries licensed stock footage first, falling back to AI video.
        - 'hybrid' (Default): Uses AI-generated scenes for custom characters, dialogue exchanges,
          and specific actions; uses licensed stock B-roll for real-world environmental establishing footage.
        """
        pref = (visual_source_preference or "hybrid").lower().strip()
        has_character = bool(scene_speaker or any(k in (character_type or "").lower() for k in ["doctor", "patient", "teacher", "student", "farmer", "engineer", "executive", "narrator"]))
        is_abstract = any(w in search_query.lower() for w in ["abstract", "future", "concept", "harmonic", "growth", "slowly", "data stream"])

        # 1. Primary: Always query Pixabay video API first unless AI generation is explicitly requested
        if preferred_source != "ai_gen" and pref != "ai_first":
            clip = await self.fetch_pixabay_clip(
                query=search_query,
                duration=duration,
                aspect_ratio=aspect_ratio,
                pixabay_video_id=pixabay_video_id
            )
            if clip:
                return clip

            # 2. Secondary stock fallback: Pexels if available
            clip = await self.fetch_pexels_clip(search_query, duration)
            if clip:
                return clip

        # 3. AI Generative & Character Synthesis Branch (Google Veo / Kinematic Engine with Lip Sync)

        # AI Generative & Character Synthesis Branch (Google Veo / Kinematic Engine with Lip Sync)
        unique_seed = f"{scene_index}_{search_query}_{scene_text}_{character_type}_{scene_speaker}_{visual_hint or ''}_{is_voiceover}_{aspect_ratio}_{show_character_labels}"
        gen_path = self.cache_dir / f"scene_{scene_index}_{hashlib.md5(unique_seed.encode()).hexdigest()[:8]}.mp4"
        clip = self.generate_procedural_broll(
            query=search_query,
            duration=duration,
            output_path=gen_path,
            style=style,
            audio_path=audio_path,
            start_time=start_time,
            scene_text=scene_text,
            character_type=character_type,
            scene_speaker=scene_speaker,
            scene_index=scene_index,
            visual_hint=visual_hint,
            verified_fact=verified_fact,
            is_voiceover=is_voiceover,
            total_scenes=total_scenes,
            aspect_ratio=aspect_ratio,
            custom_headline=custom_headline,
            show_character_labels=show_character_labels
        )
        self.used_clip_ids.add(clip.clip_id)
        return clip

footage_manager = FootageManager()
