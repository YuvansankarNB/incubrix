"""
IncuBrix Structured AI Scene Planner
Transforms raw script, topic, or audio transcript into a production-grade 13-field Scene Plan.
Fields generated for every scene:
1. Scene ID & duration
2. Narration or dialogue
3. Visual description
4. Character requirements
5. Character actions and expressions
6. Environment and background
7. Camera angle and movement
8. Video-generation prompt (Google Veo / AI Video compatible)
9. B-roll search keywords
10. Text-overlay content
11. Audio timing (start_time, end_time)
12. Required visual assets
13. Source and licensing requirements
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from src.config import settings

logger = logging.getLogger(__name__)

class ScenePlanItem:
    def __init__(
        self,
        scene_id: int,
        duration: float,
        narration: str,
        visual_description: str,
        character_requirements: str,
        character_actions: str,
        environment: str,
        camera_angle_and_movement: str,
        video_generation_prompt: str,
        b_roll_search_keywords: List[str],
        text_overlay_content: Dict[str, str],
        audio_timing: Dict[str, float],
        required_visual_assets: List[str],
        source_and_licensing_requirements: Dict[str, Any],
        speaker: Optional[str] = None
    ):
        self.scene_id = scene_id
        self.duration = duration
        self.narration = narration
        self.speaker = speaker
        self.visual_description = visual_description
        self.character_requirements = character_requirements
        self.character_actions = character_actions
        self.environment = environment
        self.camera_angle_and_movement = camera_angle_and_movement
        self.video_generation_prompt = video_generation_prompt
        self.b_roll_search_keywords = b_roll_search_keywords
        self.text_overlay_content = text_overlay_content
        self.audio_timing = audio_timing
        self.required_visual_assets = required_visual_assets
        self.source_and_licensing_requirements = source_and_licensing_requirements

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_index": self.scene_id,
            "duration": round(self.duration, 2),
            "narration": self.narration,
            "speaker": self.speaker,
            "visual_description": self.visual_description,
            "character_requirements": self.character_requirements,
            "character_actions_and_expressions": self.character_actions,
            "environment_and_background": self.environment,
            "camera_angle_and_movement": self.camera_angle_and_movement,
            "video_generation_prompt": self.video_generation_prompt,
            "b_roll_search_keywords": self.b_roll_search_keywords,
            "text_overlay_content": self.text_overlay_content,
            "audio_timing": self.audio_timing,
            "required_visual_assets": self.required_visual_assets,
            "source_and_licensing_requirements": self.source_and_licensing_requirements
        }

class ScenePlanner:
    def __init__(self):
        pass

    def plan_scenes(
        self,
        script: Optional[str] = None,
        topic: Optional[str] = None,
        scenes_input: Optional[List[Dict[str, Any]]] = None,
        total_duration: Optional[float] = None,
        character_type: str = "human",
        style: str = "bright, modern, real-world footage",
        verified_facts: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Creates a structured 13-field scene plan.
        Supports:
        - Multi-turn dialogues (e.g. Doctor: ... / Patient: ...)
        - User-written scripts
        - Structured scene list inputs
        - Topic-only input (via research facts)
        """
        # 1. Parse dialogue turns if present
        dialogue_turns = self._parse_dialogue_turns(script or "")
        
        # 2. Extract raw sentence chunks
        chunks = []
        if dialogue_turns:
            chunks = dialogue_turns
        elif scenes_input and len(scenes_input) > 0:
            for s in scenes_input:
                t = s.get("narration") or s.get("text", "")
                h = s.get("visual_hint") or ""
                chunks.append({"text": t, "hint": h, "speaker": None})
        elif script and script.strip():
            raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', script.strip()) if s.strip()]
            for s in raw_sentences:
                chunks.append({"text": s, "hint": None, "speaker": None})
        elif topic and topic.strip():
            # Topic-to-video scene synthesis
            chunks = self._synthesize_topic_chunks(topic, verified_facts)
        else:
            chunks = [{"text": "Welcome to IncuBrix AI Video Production.", "hint": "Modern AI studio", "speaker": None}]

        # 3. Calculate timing per scene
        n_scenes = len(chunks)
        est_total = total_duration if (total_duration and total_duration > 0) else max(n_scenes * 4.5, 6.0)
        scene_dur = round(est_total / n_scenes, 2)

        planned_scenes: List[Dict[str, Any]] = []
        accum_time = 0.0

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            speaker = chunk.get("speaker")
            hint = chunk.get("hint")
            start_t = round(accum_time, 2)
            end_t = round(accum_time + scene_dur, 2)
            accum_time = end_t

            # Match verified fact for this scene if available
            fact = verified_facts[i % len(verified_facts)] if (verified_facts and len(verified_facts) > 0) else None

            # Resolve character requirements and actions
            char_req, char_act, env = self._resolve_character_context(text, speaker, character_type)
            
            # Camera kinematics
            camera_angle = "Medium close-up" if (speaker or "doctor" in char_req.lower()) else "Wide establishing shot"
            camera_mov = "Slow steady push-in" if i % 2 == 0 else "Subtle lateral cinematic slider"
            camera_spec = f"{camera_angle}, {camera_mov}"

            # Visual search keywords
            keywords = self._extract_keywords(text, hint)

            # High-fidelity video generation prompt (Google Veo / AI Video compatible)
            veo_prompt = (
                f"Cinematic {camera_angle} video of {hint or text[:80]}, "
                f"set in {env}, featuring {char_req} exhibiting {char_act}, "
                f"Style: {style}, realistic volumetric lighting, smooth 24fps motion, 4k broadcast quality"
            )

            # Text overlay content
            headline = fact.get("headline") if fact else self._derive_headline(text)
            stat = fact.get("stat_number") if fact else ""
            stat_lbl = fact.get("stat_label") if fact else ""
            overlay = {
                "headline": headline,
                "stat_callout": stat,
                "stat_label": stat_lbl,
                "display_position": "negative_space_top_left",
                "face_safe": True
            }

            # Licensing requirements
            licensing = {
                "commercial_safe": True,
                "model_release_required": bool(speaker or character_type != "human"),
                "property_release_required": False,
                "allow_ai_generation": True,
                "allow_stock_footage": True,
                "permitted_licenses": ["Pexels Commercial", "Pixabay Content License", "Google Veo AI Grant"]
            }

            item = ScenePlanItem(
                scene_id=i,
                duration=scene_dur,
                narration=text,
                speaker=speaker,
                visual_description=hint or f"Detailed visualization of: {text[:90]}",
                character_requirements=char_req,
                character_actions=char_act,
                environment=env,
                camera_angle_and_movement=camera_spec,
                video_generation_prompt=veo_prompt,
                b_roll_search_keywords=keywords,
                text_overlay_content=overlay,
                audio_timing={"start_time": start_t, "end_time": end_t},
                required_visual_assets=["primary_scene_video", "synchronized_audio_track", "subtitles_overlay"],
                source_and_licensing_requirements=licensing
            )
            planned_scenes.append(item.to_dict())

        return planned_scenes

    def _parse_dialogue_turns(self, script: str) -> List[Dict[str, Any]]:
        turns = []
        lines = [line.strip() for line in script.splitlines() if line.strip()]
        pattern = re.compile(r'^\s*(?:\[|\()?([A-Za-z0-9\.\s_]{2,25}?)(?:\]|\))?\s*:\s*(.*)$')
        for line in lines:
            m = pattern.match(line)
            if m:
                spk = m.group(1).strip()
                dialogue = m.group(2).strip()
                if dialogue:
                    turns.append({"speaker": spk, "text": dialogue, "hint": f"Dialogue turn for {spk}"})
        return turns

    def _synthesize_topic_chunks(self, topic: str, verified_facts: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Constructs 3 balanced scenes for topic-to-video mode."""
        chunks = []
        chunks.append({
            "text": f"Understanding {topic} is essential for navigating today's evolving landscape.",
            "hint": f"Introduction and high-level conceptual overview of {topic}",
            "speaker": None
        })
        if verified_facts and len(verified_facts) > 0:
            f = verified_facts[0]
            chunks.append({
                "text": f"Recent authoritative research indicates that {f['fact']}",
                "hint": f"Empirical findings and key data breakdown for {topic}",
                "speaker": None
            })
        else:
            chunks.append({
                "text": f"Key principles and structured methodologies drive measurable improvements in {topic}.",
                "hint": f"Practical applications and structural implementation of {topic}",
                "speaker": None
            })
        chunks.append({
            "text": f"By adopting verified strategies in {topic}, teams can achieve sustainable long-term success.",
            "hint": f"Future outlook and positive conclusion for {topic}",
            "speaker": None
        })
        return chunks

    def _resolve_character_context(self, text: str, speaker: Optional[str], character_type: str) -> Tuple[str, str, str]:
        spk_l = (speaker or "").lower()
        txt_l = text.lower()
        ct_l = character_type.lower()

        # If a specific dialogue speaker is present, prioritize speaker matching over general scene archetype
        if spk_l:
            if "patient" in spk_l:
                return (
                    "Patient dressed in comfortable casual knitwear seated in an examination seat",
                    "Engaged listening posture, responsive head nodding, and natural eye contact",
                    "Comfortable medical examination chair in clinic room"
                )
            if "doctor" in spk_l or "physician" in spk_l:
                return (
                    "Professional clinical physician wearing a crisp white lab coat and stethoscope",
                    "Attentive listening, clear explanatory gestures, and warm reassuring expression",
                    "Bright modern clinical consultation office with anatomical diagrams"
                )
            if "student" in spk_l:
                return (
                    "Attentive student sitting at a study desk with notebook and laptop",
                    "Listening attentively, nodding thoughtfully, and taking notes",
                    "Bright modern study hall or classroom"
                )
            if "teacher" in spk_l or "professor" in spk_l or "instructor" in spk_l:
                return (
                    "Educator dressed in professional smart-casual attire",
                    "Pointing to whiteboard concepts and explaining key principles enthusiastically",
                    "Modern well-lit university seminar classroom"
                )
            if "farmer" in spk_l:
                return (
                    "Agricultural specialist wearing practical denim overalls and work shirt",
                    "Examining crop seedlings, inspecting soil health, and looking toward horizon",
                    "Lush organic farmland with green crops under sunny sky"
                )
            if "engineer" in spk_l:
                return (
                    "Principal engineer wearing clean technical attire",
                    "Demonstrating clean energy infrastructure and interacting with telemetry tablet",
                    "High-tech renewable engineering facility or wind farm"
                )
            if "presenter" in spk_l or "host" in spk_l or "anchor" in spk_l:
                return (
                    "Professional on-camera presenter in modern tailored business attire",
                    "Direct camera address, authoritative presentation gestures, and clear speech",
                    "Contemporary architectural studio with soft ambient lighting"
                )

        if "doctor" in ct_l or "physician" in txt_l or "clinic" in txt_l:
            return (
                "Professional clinical physician wearing a crisp white lab coat and stethoscope",
                "Attentive listening, clear explanatory gestures, and warm reassuring expression",
                "Bright modern clinical consultation office with anatomical diagrams"
            )
        elif "patient" in ct_l or "symptom" in txt_l:
            return (
                "Patient dressed in comfortable casual knitwear seated in an examination seat",
                "Engaged listening posture, responsive head nodding, and natural eye contact",
                "Comfortable medical examination chair in clinic room"
            )
        elif "teacher" in ct_l or "student" in txt_l or "class" in txt_l:
            return (
                "Educator dressed in professional smart-casual attire",
                "Pointing to whiteboard concepts and explaining key principles enthusiastically",
                "Modern well-lit university seminar classroom"
            )
        elif "farmer" in ct_l or "crop" in txt_l or "soil" in txt_l or "tree" in txt_l:
            return (
                "Agricultural specialist wearing practical denim overalls and work shirt",
                "Examining crop seedlings, inspecting soil health, and looking toward horizon",
                "Lush organic farmland with green crops under sunny sky"
            )
        elif "engineer" in ct_l or "tech" in txt_l or "clean energy" in txt_l:
            return (
                "Principal engineer wearing clean technical attire",
                "Demonstrating clean energy infrastructure and interacting with telemetry tablet",
                "High-tech renewable engineering facility or wind farm"
            )
        else:
            return (
                "Professional on-camera presenter in modern tailored business attire",
                "Direct camera address, authoritative presentation gestures, and clear speech",
                "Contemporary architectural studio with soft ambient lighting"
            )

    def _extract_keywords(self, text: str, hint: Optional[str]) -> List[str]:
        words = []
        if hint:
            words.extend([w.lower() for w in re.findall(r'[a-zA-Z]{3,}', hint)])
        words.extend([w.lower() for w in re.findall(r'[a-zA-Z]{4,}', text)])
        stopwords = {"this", "that", "with", "from", "were", "they", "have", "been", "which", "about", "their", "today"}
        filtered = [w for w in words if w not in stopwords]
        # Return unique top 4
        seen = set()
        res = []
        for w in filtered:
            if w not in seen:
                seen.add(w)
                res.append(w)
            if len(res) >= 4:
                break
        return res or ["cinematic", "modern", "broll", "concept"]

    def _derive_headline(self, text: str) -> str:
        words = [w for w in re.sub(r'[^a-zA-Z\s]', '', text).split() if len(w) > 3][:4]
        return " ".join(words).upper() or "KEY INSIGHT"

scene_planner = ScenePlanner()
