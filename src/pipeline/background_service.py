"""
IncuBrix Dynamic Internet Background Service
Fetches high-definition, professional photographic backgrounds relevant to ANY context:
1. Pexels API (HD/4K Commercial Photography)
2. Unsplash API (Curated High-Res Editorial Photography)
3. Pixabay API (Royalty-free Photography)
4. Wikimedia Commons High-Res Media API (verified >= 1920px real photography)
5. Intelligent semantic keyword extraction for ANY topic or visual hint
6. Cinematic Ken Burns dynamic camera push-in zoom and subtle lighting grading
"""

import os
import cv2
import json
import logging
import hashlib
import re
import urllib.request
import urllib.parse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from src.config import settings

logger = logging.getLogger(__name__)

# Curated fallback query expansions for common themes
THEME_QUERY_MAP = {
    "hospital": "modern hospital patient room interior",
    "clean_energy": "wind turbines clean energy sunset farm",
    "laboratory": "modern chemistry science laboratory microscope",
    "ecology_garden": "urban community botanical garden plants",
    "royal": "grand royal palace hall architecture",
    "finance": "modern financial stock exchange trading floor",
    "tech": "modern cloud server data center blue racks",
    "space": "deep space galaxy nebula hubble telescope",
    "culinary": "modern gourmet restaurant kitchen chef",
    "fitness": "modern athletic gym workout facility",
    "ocean": "deep blue ocean underwater coral reef",
    "automotive": "modern racetrack sports car asphalt",
    "music": "professional recording audio studio",
    "education": "modern university library lecture room",
    "architecture": "modern architectural skyscraper glass design",
    "legal": "modern court justice courtroom jurisprudence",
    "travel": "scenic alpine mountain summit sunrise",
    "social": "modern creative media studio neon",
    "warning": "modern corporate office boardroom meeting",
    "waste": "modern industrial manufacturing plant",
    "instant": "futuristic digital speed light trails",
    "default": "modern architectural interior glass daylight"
}

class BackgroundService:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("./data/cache/backgrounds")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, np.ndarray] = {}

    def extract_search_keywords(self, visual_hint: Optional[str], query: str, theme_mode: str) -> str:
        """
        Extracts clean, vivid physical search terms for any context or prompt.
        Removes filler words and punctuation, prioritizing concrete nouns and adjectives.
        """
        raw = visual_hint or query or THEME_QUERY_MAP.get(theme_mode, THEME_QUERY_MAP["default"])
        # Remove common meta phrases like "overlay", "digital metrics", "cinematic", "4k"
        cleaned = re.sub(r'\b(overlay|digital|metrics|hud|diagram|graphic|chart|concept|style|vector|illustration|4k|hd)\b', '', raw, flags=re.IGNORECASE)
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned)
        words = [w.strip() for w in cleaned.split() if len(w.strip()) > 2]
        stop_words = {"the", "and", "for", "with", "all", "are", "was", "from", "that", "this", "our", "your", "can", "will", "show", "view"}
        meaningful = [w for w in words if w.lower() not in stop_words]
        if meaningful:
            return " ".join(meaningful[:5])
        return THEME_QUERY_MAP.get(theme_mode, THEME_QUERY_MAP["default"])

    def get_background(
        self,
        query: str = "",
        visual_hint: Optional[str] = None,
        theme_mode: str = "default",
        width: int = 1920,
        height: int = 1080
    ) -> np.ndarray:
        """
        Retrieves a high-definition real photographic background matched to the scene context:
        1. Checks in-memory cache.
        2. Checks local disk cache.
        3. Queries Pexels Photo API if key available.
        4. Queries Unsplash API if key available.
        5. Queries Pixabay API if key available.
        6. Queries Open Web / Wikimedia Commons for verified high-resolution real photographs.
        7. Applies cinematic grading (photo + dark slate contrast overlay).
        """
        search_prompt = self.extract_search_keywords(visual_hint, query, theme_mode)
        cache_key = hashlib.md5(f"{theme_mode}_{search_prompt}".encode("utf-8")).hexdigest()[:12]

        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key].copy()

        disk_path = self.cache_dir / f"{cache_key}.jpg"
        if disk_path.exists():
            img = cv2.imread(str(disk_path))
            if img is not None and img.shape[0] > 200:
                graded = self._apply_cinematic_grade(img, width, height)
                self._memory_cache[cache_key] = graded
                return graded.copy()

        # Check for pre-cached theme files
        theme_fallback = self.cache_dir / f"{theme_mode}.jpg"
        if theme_fallback.exists():
            img = cv2.imread(str(theme_fallback))
            if img is not None and img.shape[0] > 200:
                graded = self._apply_cinematic_grade(img, width, height)
                self._memory_cache[cache_key] = graded
                return graded.copy()

        img_online = None

        # 1. Pexels Photo Search API
        if getattr(settings, "PEXELS_API_KEY", None):
            img_online = self._search_pexels_photo(search_prompt)

        # 2. Unsplash Search API
        if img_online is None and getattr(settings, "UNSPLASH_ACCESS_KEY", None):
            img_online = self._search_unsplash_photo(search_prompt)

        # 3. Pixabay Search API
        if img_online is None and getattr(settings, "PIXABAY_API_KEY", None):
            img_online = self._search_pixabay_photo(search_prompt)

        # 4. Open Web Wikimedia Commons High-Res Photo Search (Zero Auth)
        if img_online is None:
            img_online = self._search_open_web_photo(search_prompt)

        # Fallback to theme query if primary search yielded nothing
        if img_online is None:
            fallback_query = THEME_QUERY_MAP.get(theme_mode, "modern architectural interior daylight")
            img_online = self._search_open_web_photo(fallback_query)

        if img_online is not None:
            try:
                cv2.imwrite(str(disk_path), img_online)
            except Exception:
                pass
            graded = self._apply_cinematic_grade(img_online, width, height)
            self._memory_cache[cache_key] = graded
            return graded.copy()

        # Final procedural high-contrast fallback gradient
        fallback = self._create_procedural_gradient(theme_mode, width, height)
        self._memory_cache[cache_key] = fallback
        return fallback.copy()

    def _search_pexels_photo(self, query: str) -> Optional[np.ndarray]:
        """Queries Pexels Photos API for high-resolution landscape photography."""
        try:
            url = f"https://api.pexels.com/v1/search?query={urllib.parse.quote(query)}&per_page=5&orientation=landscape"
            req = urllib.request.Request(
                url,
                headers={"Authorization": settings.PEXELS_API_KEY, "User-Agent": "IncuBrixEngine/1.0"}
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                photos = data.get("photos", [])
                if photos:
                    img_url = photos[0].get("src", {}).get("large2x") or photos[0].get("src", {}).get("large")
                    if img_url:
                        return self._download_image(img_url)
        except Exception as e:
            logger.warning(f"Pexels photo fetch failed for '{query}': {e}")
        return None

    def _search_unsplash_photo(self, query: str) -> Optional[np.ndarray]:
        """Queries Unsplash Search API for curated editorial landscape photography."""
        try:
            url = f"https://api.unsplash.com/search/photos?query={urllib.parse.quote(query)}&orientation=landscape&per_page=5"
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Client-ID {settings.UNSPLASH_ACCESS_KEY}", "User-Agent": "IncuBrixEngine/1.0"}
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])
                if results:
                    img_url = results[0].get("urls", {}).get("regular")
                    if img_url:
                        return self._download_image(img_url)
        except Exception as e:
            logger.warning(f"Unsplash photo fetch failed for '{query}': {e}")
        return None

    def _search_pixabay_photo(self, query: str) -> Optional[np.ndarray]:
        """Queries Pixabay Image API for landscape photo."""
        try:
            url = f"https://pixabay.com/api/?key={settings.PIXABAY_API_KEY}&q={urllib.parse.quote(query)}&image_type=photo&orientation=horizontal&per_page=5"
            req = urllib.request.Request(url, headers={"User-Agent": "IncuBrixEngine/1.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                hits = data.get("hits", [])
                if hits:
                    img_url = hits[0].get("largeImageURL") or hits[0].get("webformatURL")
                    if img_url:
                        return self._download_image(img_url)
        except Exception as e:
            logger.warning(f"Pixabay photo fetch failed for '{query}': {e}")
        return None

    def _search_open_web_photo(self, query: str) -> Optional[np.ndarray]:
        """
        Queries Wikimedia Commons Open API with strict resolution filter
        (picks highest resolution true landscape image, width >= 1280).
        """
        try:
            url = (
                "https://commons.wikimedia.org/w/api.php?action=query&generator=search"
                f"&gsrsearch={urllib.parse.quote(query)}"
                "&gsrnamespace=6&gsrlimit=12&prop=imageinfo&iiprop=url|size&format=json"
            )
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "IncuBrixStudio/1.0 (contact@incubrix.ai; https://incubrix.ai)"}
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                candidates = []
                for pid, pdata in pages.items():
                    info = pdata.get("imageinfo", [{}])[0]
                    raw_url = info.get("url", "")
                    clean_url = raw_url.split("?")[0]
                    w = info.get("width", 0)
                    h = info.get("height", 0)
                    if any(clean_url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                        if w >= 1280 and w > h:
                            candidates.append((w * h, clean_url))
                
                # Sort candidates by resolution descending (highest quality first)
                candidates.sort(reverse=True)
                for _, best_url in candidates[:3]:
                    img = self._download_image(best_url)
                    if img is not None and img.shape[0] >= 500 and img.shape[1] >= 900:
                        return img
        except Exception as e:
            logger.warning(f"Open Web photographic search failed for '{query}': {e}")
        return None

    def _download_image(self, url: str) -> Optional[np.ndarray]:
        """Downloads an image from URL and decodes into a cv2 numpy array."""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=7) as resp:
                raw_bytes = resp.read()
                nparr = np.frombuffer(raw_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return img
        except Exception as e:
            logger.warning(f"Failed to download image from {url}: {e}")
            return None

    def render_cinematic_frame(
        self,
        base_bg: Optional[np.ndarray],
        frame_idx: int,
        total_frames: int,
        width: int = 1920,
        height: int = 1080
    ) -> np.ndarray:
        """
        Applies a subtle, elegant Ken Burns push-in zoom effect (1.00x to 1.05x)
        across the scene duration so background footage is alive and cinematic.
        """
        if base_bg is None or base_bg.size == 0:
            return self._create_procedural_gradient("default", width, height)

        progress = max(0.0, min(1.0, frame_idx / max(1, total_frames)))
        # Subtle 5% zoom over duration
        zoom = 1.0 + 0.05 * progress
        
        orig_h, orig_w = base_bg.shape[:2]
        crop_w = int(orig_w / zoom)
        crop_h = int(orig_h / zoom)
        
        x_start = (orig_w - crop_w) // 2
        y_start = (orig_h - crop_h) // 2
        
        cropped = base_bg[y_start:y_start + crop_h, x_start:x_start + crop_w]
        frame = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        return frame

    def _apply_cinematic_grade(self, img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Crops/resizes to target aspect ratio and applies a subtle 75/25 dark slate overlay."""
        h, w = img.shape[:2]
        target_aspect = target_w / target_h
        current_aspect = w / h

        if current_aspect > target_aspect:
            new_w = int(h * target_aspect)
            x_start = (w - new_w) // 2
            cropped = img[:, x_start:x_start + new_w]
        else:
            new_h = int(w / target_aspect)
            y_start = (h - new_h) // 2
            cropped = img[y_start:y_start + new_h, :]

        # Scale slightly larger than target for clean Ken Burns zoom
        scaled_w = int(target_w * 1.1)
        scaled_h = int(target_h * 1.1)
        resized = cv2.resize(cropped, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

        # 75% photo + 25% dark slate overlay so HUD graphics and text stand out with optimal contrast
        overlay = np.zeros_like(resized)
        overlay[:] = (22, 26, 32)
        cv2.addWeighted(resized, 0.75, overlay, 0.25, 0, resized)
        return resized

    def _create_procedural_gradient(self, theme_mode: str, width: int, height: int) -> np.ndarray:
        """High-contrast fallback dark gradient if network is unavailable."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        base_colors = {
            "hospital": (42, 35, 30),
            "clean_energy": (30, 40, 25),
            "laboratory": (45, 38, 25),
            "ecology_garden": (25, 45, 30),
            "royal": (35, 20, 45),
            "finance": (40, 35, 25),
            "tech": (45, 25, 20),
            "space": (25, 20, 25),
            "default": (35, 30, 28)
        }
        col = base_colors.get(theme_mode, base_colors["default"])
        canvas[:] = col
        for y in range(height):
            factor = 0.8 + 0.4 * (y / height)
            canvas[y, :] = (int(col[0] * factor), int(col[1] * factor), int(col[2] * factor))
        return canvas

background_service = BackgroundService()
