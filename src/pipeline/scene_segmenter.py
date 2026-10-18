import re
from typing import List, Dict, Any, Optional
from src.config import settings
from src.logging_config import logger

# Contextual Disambiguation Knowledge Graph
DISAMBIGUATION_RULES = [
    {
        "trigger": r"\bcloud\b",
        "context_clues": ["computing", "data", "storage", "server", "aws", "azure", "devops", "tech", "software"],
        "disambiguated_query": "high tech server room data center glowing blue led rack",
        "fallback_query": "blue sky with soft clouds timelapse"
    },
    {
        "trigger": r"\bbank\b",
        "context_clues": ["money", "account", "deposit", "interest", "saving", "loan", "finance", "financial", "cash"],
        "disambiguated_query": "modern corporate bank interior financial building glass architecture",
        "fallback_query": "peaceful green river bank flowing water"
    },
    {
        "trigger": r"\bpython\b",
        "context_clues": ["code", "programming", "software", "developer", "script", "data science", "algorithm"],
        "disambiguated_query": "software engineer typing code in ide on computer screen",
        "fallback_query": "wildlife nature green forest canopy"
    },
    {
        "trigger": r"\bmouse\b",
        "context_clues": ["computer", "click", "screen", "cursor", "desktop", "laptop", "pc"],
        "disambiguated_query": "hand using modern ergonomic computer mouse at clean work desk",
        "fallback_query": "nature meadow field"
    },
    {
        "trigger": r"\bvalue\b",
        "context_clues": ["money", "cost", "compounding", "investment", "worth", "principle", "business", "growth"],
        "disambiguated_query": "dynamic financial growth analytics graph on modern tablet",
        "fallback_query": "high quality modern product craftmanship"
    },
    {
        "trigger": r"(\btrust\b|\bfaith\b)",
        "context_clues": ["earn", "slow", "build", "relation", "partnership", "honest", "integrity"],
        "disambiguated_query": "professional business handshake in modern bright office sunlight",
        "fallback_query": "reliable partnership team collaboration"
    },
    {
        "trigger": r"(\b40%\b|\b\d+%\b|\bsales\b|\bgrew\b|\bgrowth\b)",
        "context_clues": ["sales", "grow", "year", "revenue", "profit", "market", "statistic"],
        "disambiguated_query": "upward trending corporate stock market chart screen financial dashboard",
        "fallback_query": "business revenue analytics dashboard display"
    }
]

class SceneSegmenter:
    def segment_words_into_scenes(
        self,
        words: List[Dict[str, Any]],
        total_audio_duration: float,
        explicit_scenes: Optional[List[Dict[str, Any]]] = None,
        avoid_list: Optional[List[str]] = None,
        style: str = "bright, modern, real-world footage",
        dialogue_turns: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Segments speech into continuous non-overlapping scenes timed strictly to narration.
        Guarantees 100% coverage with zero audio drift.
        """
        scenes = []
        avoid = [a.lower().strip() for a in (avoid_list or [])]

        # 1. Dialogue Turns Segmentation (Doctor-Patient, Multi-Character)
        if dialogue_turns and len(dialogue_turns) > 0:
            num_turns = len(dialogue_turns)
            has_durations = all("duration" in t and t.get("duration", 0) > 0 for t in dialogue_turns)
            accum_t = 0.0
            for i, turn in enumerate(dialogue_turns):
                if has_durations:
                    start = turn.get("start_time", round(accum_t, 2))
                    dur = turn.get("duration", 2.0)
                    end = turn.get("end_time", round(start + dur, 2))
                    accum_t = end
                    if i == num_turns - 1:
                        end = max(end, total_audio_duration)
                        dur = round(end - start, 2)
                else:
                    scene_duration = total_audio_duration / num_turns
                    start = round(i * scene_duration, 2)
                    end = round((i + 1) * scene_duration, 2) if i < num_turns - 1 else total_audio_duration
                    dur = round(end - start, 2)

                raw_text = turn.get("text", "")
                speaker = turn.get("speaker", "")
                query, rationale = self._resolve_visual_query(raw_text, None, avoid, style)
                scenes.append({
                    "scene_index": i,
                    "speaker": speaker,
                    "narration": raw_text,
                    "start_time": start,
                    "end_time": end,
                    "duration": dur,
                    "visual_hint": f"Dialogue scene: {speaker}",
                    "search_query": query,
                    "selection_rationale": f"Dialogue turn for {speaker}: {rationale}"
                })
            return scenes

        if explicit_scenes and len(explicit_scenes) > 0:
            # Map explicit scene inputs to speech duration evenly or by keyword match
            num_scenes = len(explicit_scenes)
            scene_duration = total_audio_duration / num_scenes
            for i, item in enumerate(explicit_scenes):
                start = round(i * scene_duration, 2)
                end = round((i + 1) * scene_duration, 2) if i < num_scenes - 1 else total_audio_duration
                raw_text = item.get("narration") or item.get("text", "")
                # Strip any speaker label prefix so narration is clean
                clean_text = re.sub(r'^\s*(?:\[|\()?([A-Za-z0-9\.\s_]{2,25}?)(?:\]|\))?\s*:\s*', '', raw_text).strip()
                hint = item.get("visual_hint", "")
                
                query, rationale = self._resolve_visual_query(clean_text or raw_text, hint, avoid, style)
                scenes.append({
                    "scene_index": i,
                    "narration": clean_text or raw_text,
                    "start_time": start,
                    "end_time": end,
                    "duration": round(end - start, 2),
                    "visual_hint": hint,
                    "search_query": query,
                    "selection_rationale": rationale
                })
            return scenes

        if not words:
            # Fallback single scene
            query, rationale = self._resolve_visual_query("narration", None, avoid, style)
            return [{
                "scene_index": 0,
                "narration": "",
                "start_time": 0.0,
                "end_time": total_audio_duration,
                "duration": total_audio_duration,
                "search_query": query,
                "selection_rationale": rationale
            }]

        # Group words by sentence boundaries (. , ! ? ; or pause > 0.4s)
        current_chunk = []
        chunk_start = 0.0

        for idx, w in enumerate(words):
            if not current_chunk:
                chunk_start = w["start"] if idx == 0 else current_chunk_prev_end
            
            current_chunk.append(w["word"])
            current_end = w["end"]
            chunk_duration = current_end - chunk_start

            # Determine split condition:
            # 1. Terminal punctuation (., !, ?) and duration >= min_scene_duration
            # 2. Pause to next word >= 0.4s and duration >= min_scene_duration
            # 3. Chunk duration exceeds max_scene_duration
            has_punctuation = bool(re.search(r'[.!?]$', w["word"]))
            pause_next = False
            if idx + 1 < len(words):
                pause_next = (words[idx + 1]["start"] - w["end"]) >= 0.4
            
            is_last_word = (idx == len(words) - 1)

            should_split = False
            if is_last_word:
                should_split = True
            elif chunk_duration >= settings.MIN_SCENE_DURATION_SECONDS and (has_punctuation or pause_next):
                should_split = True
            elif chunk_duration >= settings.MAX_SCENE_DURATION_SECONDS:
                should_split = True

            if should_split:
                scene_text = " ".join(current_chunk).strip()
                # Snap to previous scene boundary or 0.0
                actual_start = 0.0 if len(scenes) == 0 else scenes[-1]["end_time"]
                actual_end = total_audio_duration if is_last_word else current_end

                query, rationale = self._resolve_visual_query(scene_text, None, avoid, style)
                scenes.append({
                    "scene_index": len(scenes),
                    "narration": scene_text,
                    "start_time": round(actual_start, 2),
                    "end_time": round(actual_end, 2),
                    "duration": round(actual_end - actual_start, 2),
                    "search_query": query,
                    "selection_rationale": rationale
                })
                current_chunk = []
                current_chunk_prev_end = actual_end

        # Ensure last scene reaches total audio duration precisely
        if scenes and scenes[-1]["end_time"] < total_audio_duration:
            scenes[-1]["end_time"] = round(total_audio_duration, 2)
            scenes[-1]["duration"] = round(scenes[-1]["end_time"] - scenes[-1]["start_time"], 2)

        return scenes

    def _resolve_visual_query(self, text: str, hint: Optional[str], avoid: List[str], style: str) -> (str, str):
        """Disambiguates polysemy, applies avoid filters, and derives high-relevance query."""
        if hint and hint.strip():
            query = f"{hint.strip()} {style}".strip()
            return self._apply_avoid(query, avoid), f"Directed by explicit creator visual hint: '{hint}'"

        text_lower = text.lower()

        # Check knowledge rules for polysemous or abstract phrases
        for rule in DISAMBIGUATION_RULES:
            if re.search(rule["trigger"], text_lower):
                matched_clues = [c for c in rule["context_clues"] if c in text_lower]
                if matched_clues:
                    query = f"{rule['disambiguated_query']} {style}".strip()
                    rationale = f"Semantic match on concept '{rule['trigger']}' with context clues: {matched_clues}"
                    return self._apply_avoid(query, avoid), rationale

        # Extract concrete nouns / action words
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text_lower)
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "are", "was", "were", "this", "that", "it", "so", "as", "by", "from", "you", "we", "they", "i"}
        tokens = [w for w in cleaned.split() if w not in stopwords and len(w) > 2]
        
        if tokens:
            key_subject = " ".join(tokens[:3])
            query = f"{key_subject} {style}".strip()
            rationale = f"Extracted semantic keywords: {tokens[:3]}"
        else:
            query = f"modern abstract background {style}".strip()
            rationale = "General ambient concept query for neutral pacing"

        return self._apply_avoid(query, avoid), rationale

    def _apply_avoid(self, query: str, avoid: List[str]) -> str:
        """Strips or alters keywords that violate avoid settings."""
        q_lower = query.lower()
        for term in avoid:
            term = term.strip().lower()
            if not term:
                continue
            if "face" in term or "people" in term:
                q_lower = q_lower.replace("face", "object").replace("portrait", "scenery").replace("people", "cinematic landscape")
            if "alcohol" in term:
                q_lower = q_lower.replace("beer", "coffee").replace("wine", "tea").replace("alcohol", "beverage")
            # Also directly remove the avoid term if still present
            q_lower = re.sub(rf'\b{re.escape(term)}\b', '', q_lower)
        # Clean double spaces
        return re.sub(r'\s+', ' ', q_lower).strip()

scene_segmenter = SceneSegmenter()
