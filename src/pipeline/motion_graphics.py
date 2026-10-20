"""
IncuBrix Professional Motion Graphics & Intelligent Typography Engine
Replaces generic software dashboard boxes with context-aware broadcast-grade video typography.
Features:
1. Intelligent Negative Space Calculation:
   - Detects/estimates character bounding boxes & protected areas (heads, faces, bodies).
   - Protects lower-third subtitle area from collisions.
   - Constrains overlays strictly into free negative space (e.g. left side when presenter is on right).
   - In dialogue/conversation scenes, lets characters breathe with zero intrusive overlay boxes.
2. Value-Additive Typography:
   - Removes artificial diagnostics ("VERIFIED METRIC DATA", "DRAMATIC CONVERSATION", "KEY SUMMARY TAKEAWAY").
   - Displays clean, contextual headlines, verified metrics with authentic source attribution, or concise key terms.
   - Never overlaps character heads or faces.
"""

import cv2
import math
import re
import numpy as np
from typing import Optional, Dict, Any, Tuple

# Coordinated Theme Accent Palettes (BGR)
THEME_PALETTES = {
    "medical": {"primary": (230, 180, 50), "accent": (240, 220, 100), "bg": (20, 16, 28)},
    "environmental": {"primary": (80, 215, 120), "accent": (140, 245, 180), "bg": (18, 26, 20)},
    "education": {"primary": (50, 180, 245), "accent": (120, 220, 255), "bg": (28, 22, 18)},
    "tech": {"primary": (240, 140, 50), "accent": (255, 190, 100), "bg": (26, 18, 20)},
    "business": {"primary": (220, 130, 60), "accent": (240, 180, 120), "bg": (22, 20, 26)},
    "default": {"primary": (235, 160, 50), "accent": (245, 200, 100), "bg": (20, 20, 26)}
}

class MotionGraphicsEngine:
    def __init__(self):
        pass

    def detect_theme(self, text: str) -> str:
        """Detects high-level topic theme for color harmony."""
        t_low = (text or "").lower()
        if any(w in t_low for w in ["doctor", "patient", "clinic", "hospital", "health", "medicine", "symptom", "blood", "vitals", "recovery"]):
            return "medical"
        if any(w in t_low for w in ["tree", "trees", "canopy", "garden", "plant", "green", "carbon", "wind", "renewable", "soil", "farmer", "eco", "sustain", "environment"]):
            return "environmental"
        if any(w in t_low for w in ["teach", "teacher", "student", "learn", "study", "retention", "school", "education", "lesson", "class"]):
            return "education"
        if any(w in t_low for w in ["code", "software", "ai", "cloud", "server", "algorithm", "data", "developer", "engineer", "tech", "velocity"]):
            return "tech"
        if any(w in t_low for w in ["business", "finance", "revenue", "sales", "executive", "company", "market", "roi", "client", "profit"]):
            return "business"
        return "default"

    def should_render_overlay(
        self,
        scene_text: str,
        character_pos: str,
        verified_fact: Optional[Dict[str, Any]] = None,
        scene_index: int = 0,
        total_scenes: int = 1
    ) -> bool:
        """
        Determines whether an informative overlay graphic adds genuine value to the scene:
        1. In dialogue/conversation scenes with multiple speakers: return False.
           Characters and natural dialogue take priority; subtitles communicate speech.
        2. In emotional / dramatic scenes: return False.
        3. In educational / explainer / statistical scenes: return True only if there is
           a verified fact, metric, process, or concise key term to highlight.
        """
        # Dialogue scenes: Let characters breathe; zero overlay boxes
        if character_pos == "dialogue":
            return False

        t_low = (scene_text or "").lower()

        # Scenes focused on personal emotion, greetings, or casual conversation
        emotional_words = [
            "thank you", "thanks", "hello", "goodbye", "worried", "scared", "hope", 
            "relief", "glad", "feel", "appreciate", "welcome", "good morning", "honest with me"
        ]
        if any(w in t_low for w in emotional_words) and not verified_fact:
            return False

        # If a verified fact is present with concrete numbers or metrics:
        if verified_fact and (verified_fact.get("stat_number") or verified_fact.get("metric")):
            return True

        # If the script narration directly discusses numbers, metrics, or research:
        if any(w in t_low for w in ["%", "percent", "metric", "rate", "increased by", "reduced by", "doubled", "measure", "ratio"]):
            return True

        # Educational explainer: concept definitions or multi-step processes
        if any(w in t_low for w in ["first", "step", "process", "workflow", "concept", "means", "defined as", "principle", "rule"]):
            return True

        # Opening scene of single-presenter explainer or closing summary
        if scene_index == 0 or scene_index == max(1, total_scenes - 1):
            return True

        return False

    def select_graphic_type(
        self,
        scene_text: str,
        scene_index: int,
        total_scenes: int,
        verified_fact: Optional[Dict[str, Any]] = None
    ) -> str:
        """Automatically selects the best motion graphic type according to scene context."""
        t_low = (scene_text or "").lower()

        # 1. Concrete metric or verified research fact
        if (verified_fact and (verified_fact.get("stat_number") or verified_fact.get("metric"))) or \
           any(w in t_low for w in ["%", "percent", "rate", "increased by", "reduced by", "doubled"]):
            return "statistic"
        
        # 2. Structural scene positions
        if scene_index == 0:
            return "opening_title"
        if scene_index == max(1, total_scenes - 1):
            return "closing_message"
        
        # 3. Process or workflow
        if any(w in t_low for w in ["first", "then", "next", "finally", "step", "process", "workflow", "phase"]):
            return "process"
        
        # 4. Comparison
        if any(w in t_low for w in ["versus", "vs", "compared to", "before", "after", "traditional", "instead of"]):
            return "comparison"
        
        # 5. Key phrase / Concept definition
        return "key_phrase"

    def render_motion_graphics(
        self,
        canvas: np.ndarray,
        scene_text: str,
        t: float,
        duration: float,
        scene_index: int,
        total_scenes: int,
        character_pos: str = "right",
        aspect_ratio: str = "16:9",
        verified_fact: Optional[Dict[str, Any]] = None,
        custom_headline: Optional[str] = None
    ):
        """
        Renders an intelligent, beautifully animated motion graphic onto the scene canvas.
        Calculates negative space so text never collides with characters or faces.
        """
        # Value-Add Check: Skip overlay if not appropriate (e.g. dialogue or casual scenes)
        if not self.should_render_overlay(
            scene_text=scene_text,
            character_pos=character_pos,
            verified_fact=verified_fact,
            scene_index=scene_index,
            total_scenes=total_scenes
        ):
            return

        h, w = canvas.shape[:2]
        theme = self.detect_theme(scene_text)
        palette = THEME_PALETTES.get(theme, THEME_PALETTES["default"])
        g_type = self.select_graphic_type(scene_text, scene_index, total_scenes, verified_fact)

        # Time-based animation progress
        progress = max(0.0, min(1.0, t / max(0.1, duration)))
        
        # Smooth entry ease (0.0 to 0.18 relative progress)
        if progress < 0.18:
            ease = math.sin((progress / 0.18) * (math.pi / 2.0))
            slide_offset = int((1.0 - ease) * 35)
            alpha_mult = ease
        elif progress > 0.85:
            # Smooth exit ease
            exit_ease = (1.0 - progress) / 0.15
            slide_offset = 0
            alpha_mult = max(0.0, exit_ease)
        else:
            slide_offset = 0
            alpha_mult = 1.0

        if alpha_mult <= 0.02:
            return

        # ----------------- NEGATIVE SPACE & ZERO OVERLAP GEOMETRY -----------------
        # Protected Zones:
        # 1. Subtitles safe area: y in [int(h * 0.80), h]
        # 2. Character safe area:
        #    - If character on right: x in [int(w * 0.52), w], y in [int(h * 0.10), h]
        #    - If character on left:  x in [0, int(w * 0.48)], y in [int(h * 0.10), h]
        #    - If portrait 9:16:      x in [0, w], y in [int(h * 0.38), h]
        if h > w:
            # Actual vertical portrait canvas (e.g. 1080x1920)
            card_w = int(w * 0.84)
            card_x = (w - card_w) // 2
            card_y = int(h * 0.08) + slide_offset
            card_h = min(int(h * 0.18), 180)
        elif character_pos == "right":
            # Character is on right (cx ≈ 0.78 * w).
            # Confine overlay strictly to left negative space: x in [0.06 * w, 0.48 * w]
            card_w = min(int(w * 0.42), 520)
            card_x = int(w * 0.06) - slide_offset
            card_y = int(h * 0.20)
            card_h = min(int(h * 0.46), 250)
        elif character_pos == "left":
            # Character is on left (cx ≈ 0.28 * w).
            # Confine overlay strictly to right negative space: x in [0.52 * w, 0.94 * w]
            card_w = min(int(w * 0.42), 520)
            card_x = int(w * 0.52) + slide_offset
            card_y = int(h * 0.20)
            card_h = min(int(h * 0.46), 250)
        elif aspect_ratio == "9:16":
            # Raw horizontal frame generated for vertical output: keep card strictly in left negative space
            card_w = min(int(w * 0.42), 520)
            card_x = int(w * 0.06)
            card_y = int(h * 0.20)
            card_h = min(int(h * 0.44), 230)
        else:
            # Balanced center-left
            card_w = min(int(w * 0.46), 560)
            card_x = int(w * 0.08) + slide_offset
            card_y = int(h * 0.22)
            card_h = min(int(h * 0.44), 250)

        # Create overlay layer for smooth alpha blending
        overlay = canvas.copy()

        # ----------------- GRAPHIC TYPE IMPLEMENTATIONS -----------------
        if g_type == "opening_title":
            self._draw_opening_title(
                overlay, card_x, card_y, card_w, card_h, scene_text, 
                theme, palette, custom_headline, t
            )
        elif g_type == "statistic":
            self._draw_statistic_card(
                overlay, card_x, card_y, card_w, card_h, scene_text, 
                verified_fact, palette, t, progress
            )
        elif g_type == "process":
            self._draw_process_card(
                overlay, card_x, card_y, card_w, card_h, scene_text, 
                palette, progress
            )
        elif g_type == "comparison":
            self._draw_comparison_card(
                overlay, card_x, card_y, card_w, card_h, scene_text, 
                palette
            )
        elif g_type == "closing_message":
            self._draw_closing_card(
                overlay, card_x, card_y, card_w, card_h, scene_text, 
                palette, t
            )
        else: # "key_phrase"
            self._draw_key_phrase_card(
                overlay, card_x, card_y, card_w, card_h, scene_text, 
                palette, verified_fact, t
            )

        # Blend overlay onto canvas according to alpha_mult
        cv2.addWeighted(overlay, alpha_mult, canvas, 1.0 - alpha_mult, 0, canvas)

    def _draw_glass_panel(self, canvas, x, y, w, h, border_color=(70, 85, 110)):
        """Draws a clean, softly shaded glassmorphic card with subtle accent border."""
        # Glass fill
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (14, 18, 25), -1)
        # Subtle top accent line
        cv2.rectangle(canvas, (x, y), (x + w, y + 3), border_color, -1)
        # Clean 1px border
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1, cv2.LINE_AA)

    def _draw_opening_title(self, canvas, x, y, w, h, text, theme, palette, custom_headline, t):
        self._draw_glass_panel(canvas, x, y, w, h, palette["primary"])
        
        # Clean Eyebrow
        eyebrow = f"{theme.upper()} OVERVIEW"
        cv2.putText(canvas, eyebrow, (x + 24, y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.46, palette["accent"], 1, cv2.LINE_AA)

        # Headline
        headline = (custom_headline or self._clean_headline(text, 4)).upper()
        cv2.putText(canvas, headline, (x + 24, y + 80), cv2.FONT_HERSHEY_DUPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)

        # Accent pulse line
        line_w = int(min(w - 48, (w - 48) * min(1.0, t * 1.5)))
        cv2.line(canvas, (x + 24, y + 98), (x + 24 + line_w, y + 98), palette["primary"], 2, cv2.LINE_AA)

        # Concise subtitle takeaway
        sub = self._wrap_text(text, 36)
        for i, s_line in enumerate(sub[:2]):
            cv2.putText(canvas, s_line, (x + 24, y + 138 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (200, 210, 225), 1, cv2.LINE_AA)

    def _sanitize_text(self, text: str) -> str:
        if not text:
            return ""
        return (
            str(text).replace("°C", "C")
            .replace("°", " deg")
            .replace("—", "-")
            .replace("–", "-")
            .replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("•", "*")
        )

    def _draw_statistic_card(self, canvas, x, y, w, h, text, fact, palette, t, progress):
        self._draw_glass_panel(canvas, x, y, w, h, palette["accent"])
        
        # Extract authentic stat number & label from verified fact or text
        stat_num = fact.get("stat_number") if fact else None
        if not stat_num:
            stat_num = self._extract_number_from_text(text)
        stat_num = self._sanitize_text(stat_num)

        stat_lbl = fact.get("stat_label") if fact else None
        if not stat_lbl:
            stat_lbl = "Measured Impact"
        stat_lbl = self._sanitize_text(stat_lbl)

        publisher = fact.get("publisher") if fact else None

        # Clean Topic Eyebrow
        eyebrow = fact.get("topic", "KEY METRIC").upper() if fact else "KEY METRIC"
        cv2.putText(canvas, eyebrow, (x + 24, y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.44, (160, 175, 195), 1, cv2.LINE_AA)

        # Beautiful Numeric Callout
        cv2.putText(canvas, stat_num, (x + 24, y + 98), cv2.FONT_HERSHEY_DUPLEX, 1.45, palette["accent"], 2, cv2.LINE_AA)

        # Metric Description
        cv2.putText(canvas, stat_lbl.upper(), (x + 24, y + 134), cv2.FONT_HERSHEY_DUPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)

        # Supporting text line
        sub = self._wrap_text(fact.get("fact", text) if fact else text, 36)
        if sub and h > 190:
            cv2.putText(canvas, sub[0], (x + 24, y + 168), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (190, 205, 220), 1, cv2.LINE_AA)

        # Authentic Source Attribution Citation Pill
        if publisher and h > 220:
            src_text = f"Source: {publisher}"
            cv2.rectangle(canvas, (x + 20, y + h - 36), (x + w - 20, y + h - 10), (24, 30, 42), -1)
            cv2.rectangle(canvas, (x + 20, y + h - 36), (x + w - 20, y + h - 10), (60, 75, 100), 1, cv2.LINE_AA)
            cv2.circle(canvas, (x + 32, y + h - 23), 3, palette["primary"], -1, cv2.LINE_AA)
            cv2.putText(canvas, src_text, (x + 42, y + h - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 215, 230), 1, cv2.LINE_AA)

    def _draw_process_card(self, canvas, x, y, w, h, text, palette, progress):
        self._draw_glass_panel(canvas, x, y, w, h, palette["primary"])
        cv2.putText(canvas, "PROCESS FLOW", (x + 24, y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.46, palette["accent"], 1, cv2.LINE_AA)

        # 3 Step Nodes: Input -> Process -> Outcome
        steps = ["Preparation", "Execution", "Outcome"]
        node_w = (w - 60) // 3
        active_step = min(2, int(progress * 3.5))

        for i, s_name in enumerate(steps):
            nx = x + 20 + i * node_w
            ny = y + 68
            is_active = (i == active_step)
            box_col = (30, 40, 55) if not is_active else (40, 60, 85)
            border_col = (80, 95, 115) if not is_active else palette["primary"]

            cv2.rectangle(canvas, (nx, ny), (nx + node_w - 10, ny + 65), box_col, -1)
            cv2.rectangle(canvas, (nx, ny), (nx + node_w - 10, ny + 65), border_col, 2 if is_active else 1, cv2.LINE_AA)

            # Step number
            cv2.circle(canvas, (nx + 16, ny + 22), 9, border_col, -1, cv2.LINE_AA)
            cv2.putText(canvas, str(i + 1), (nx + 13, ny + 26), cv2.FONT_HERSHEY_DUPLEX, 0.38, (15, 15, 20), 1, cv2.LINE_AA)
            cv2.putText(canvas, s_name, (nx + 10, ny + 50), cv2.FONT_HERSHEY_DUPLEX, 0.40, (240, 240, 250), 1, cv2.LINE_AA)

            # Connecting arrow
            if i < 2:
                arrow_x = nx + node_w - 10
                cv2.line(canvas, (arrow_x, ny + 32), (arrow_x + 10, ny + 32), palette["primary"], 2, cv2.LINE_AA)

        # Context line below
        sub = self._wrap_text(text, 36)
        if sub and h > 180:
            cv2.putText(canvas, sub[0], (x + 24, y + 172), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 215, 230), 1, cv2.LINE_AA)

    def _draw_comparison_card(self, canvas, x, y, w, h, text, palette):
        self._draw_glass_panel(canvas, x, y, w, h, (100, 115, 140))
        cv2.putText(canvas, "COMPARISON", (x + 24, y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.46, (170, 185, 205), 1, cv2.LINE_AA)

        col_w = (w - 60) // 2
        cy1 = y + 62
        # Col 1: Traditional
        cx1 = x + 20
        cv2.rectangle(canvas, (cx1, cy1), (cx1 + col_w, cy1 + 95), (32, 22, 26), -1)
        cv2.rectangle(canvas, (cx1, cy1), (cx1 + col_w, cy1 + 95), (70, 50, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "CONVENTIONAL", (cx1 + 14, cy1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.46, (100, 110, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "High Friction", (cx1 + 14, cy1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 210), 1, cv2.LINE_AA)

        # Col 2: Modern
        cx2 = x + 35 + col_w
        cv2.rectangle(canvas, (cx2, cy1), (cx2 + col_w, cy1 + 95), (20, 34, 28), -1)
        cv2.rectangle(canvas, (cx2, cy1), (cx2 + col_w, cy1 + 95), palette["primary"], 1, cv2.LINE_AA)
        cv2.putText(canvas, "OPTIMIZED", (cx2 + 14, cy1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.46, palette["accent"], 1, cv2.LINE_AA)
        cv2.putText(canvas, "Zero-Lag Flow", (cx2 + 14, cy1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 255, 230), 1, cv2.LINE_AA)

    def _draw_closing_card(self, canvas, x, y, w, h, text, palette, t):
        self._draw_glass_panel(canvas, x, y, w, h, palette["accent"])
        cv2.putText(canvas, "SUMMARY", (x + 24, y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.46, palette["primary"], 1, cv2.LINE_AA)
        
        headline = self._clean_headline(text, 4).upper()
        cv2.putText(canvas, headline, (x + 24, y + 80), cv2.FONT_HERSHEY_DUPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)

        sub = self._wrap_text(text, 36)
        for i, s_line in enumerate(sub[:2]):
            cv2.putText(canvas, s_line, (x + 24, y + 130 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (210, 225, 240), 1, cv2.LINE_AA)

    def _draw_key_phrase_card(self, canvas, x, y, w, h, text, palette, fact, t):
        self._draw_glass_panel(canvas, x, y, w, h, palette["primary"])
        cv2.putText(canvas, "KEY CONCEPT", (x + 24, y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.46, palette["accent"], 1, cv2.LINE_AA)

        headline = self._clean_headline(text, 4).upper()
        cv2.putText(canvas, headline, (x + 24, y + 80), cv2.FONT_HERSHEY_DUPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)

        sub = self._wrap_text(text, 36)
        for i, s_line in enumerate(sub[:2]):
            cv2.putText(canvas, s_line, (x + 24, y + 130 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (200, 215, 230), 1, cv2.LINE_AA)

    def _clean_headline(self, text: str, max_words: int = 4) -> str:
        words = [w for w in re.sub(r'[^a-zA-Z0-9\s]', '', text).split() if len(w) > 2 and w.lower() not in {"the", "and", "for", "with", "this", "that", "our", "your", "are", "was"}]
        if words:
            return " ".join(words[:max_words])
        return "ESSENTIAL INSIGHT"

    def _extract_number_from_text(self, text: str) -> str:
        match = re.search(r'([+-]?\d+(?:\.\d+)?%|\$\d+(?:\.\d+)?[MBK]?|\d+x|\d+(?:,\d+)?)', text)
        if match:
            return match.group(1)
        return "KEY METRIC"

    def _wrap_text(self, text: str, max_chars: int = 35) -> list:
        words = text.split()
        lines = []
        cur = []
        cur_len = 0
        for w in words:
            if cur_len + len(w) + 1 > max_chars:
                lines.append(" ".join(cur))
                cur = [w]
                cur_len = len(w)
            else:
                cur.append(w)
                cur_len += len(w) + 1
        if cur:
            lines.append(" ".join(cur))
        return lines

motion_graphics = MotionGraphicsEngine()
