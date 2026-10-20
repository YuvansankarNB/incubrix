"""
IncuBrix Pixabay Video Provider
================================
A dedicated, production-grade adapter for the official Pixabay Video API.

Features:
- Keyword search with ranked, relevance-scored results
- Synonym-based fallback (3 retry tiers)
- Duplicate-prevention per job
- Rate-limit handling with bounded exponential back-off (max 3 retries)
- Graceful degradation to mock mode when no API key is set
- Strict proxy pattern: API key NEVER leaves the server process
- Full provenance metadata matching Pixabay Content License requirements
"""

import re
import os
import json
import math
import time
import uuid
import hashlib
import logging
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PixabayVideoResult:
    """Full provenance + quality record for a single Pixabay video hit."""
    pixabay_id: int
    page_url: str
    user: str
    user_id: int
    user_profile_url: str
    tags: str
    duration: int                  # seconds (from API)
    thumbnail_url: str
    download_url: str
    width: int
    height: int
    size: int                      # bytes
    quality_tier: str              # "large" | "medium" | "small" | "tiny"
    editors_choice: bool
    views: int
    downloads: int
    relevance_score: float = 0.0
    is_mock: bool = False

    @property
    def aspect_ratio_float(self) -> float:
        return self.width / self.height if self.height else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pixabay_id": self.pixabay_id,
            "page_url": self.page_url,
            "user": self.user,
            "user_id": self.user_id,
            "user_profile_url": self.user_profile_url,
            "tags": self.tags,
            "duration": self.duration,
            "thumbnail_url": self.thumbnail_url,
            "download_url": self.download_url,
            "width": self.width,
            "height": self.height,
            "size": self.size,
            "quality_tier": self.quality_tier,
            "editors_choice": self.editors_choice,
            "views": self.views,
            "downloads": self.downloads,
            "relevance_score": round(self.relevance_score, 3),
            "is_mock": self.is_mock,
        }


# ---------------------------------------------------------------------------
# Synonym / fallback query expansion table
# ---------------------------------------------------------------------------

_SYNONYM_MAP: Dict[str, List[str]] = {
    "doctor": ["physician", "medical professional", "healthcare worker"],
    "patient": ["person clinic", "medical visit", "hospital room"],
    "medical": ["healthcare", "clinical", "hospital"],
    "teacher": ["educator", "instructor", "classroom teaching"],
    "student": ["learner", "study", "academic"],
    "farmer": ["agriculture", "farm field", "rural"],
    "engineer": ["technology", "technical work", "innovation"],
    "business": ["office professional", "corporate", "workplace"],
    "nature": ["landscape", "outdoor environment", "scenic"],
    "city": ["urban", "street", "metropolitan"],
    "science": ["laboratory", "research", "experiment"],
    "coding": ["programming", "software development", "computer"],
    "food": ["cuisine", "cooking", "meal"],
    "exercise": ["fitness", "workout", "sport"],
    "education": ["learning", "school", "training"],
    "environment": ["ecology", "conservation", "sustainability"],
}


def _generate_synonyms(query: str, tier: int = 0) -> List[str]:
    """
    Returns a list of progressively broader synonym queries for the given tier.
    tier=0 → original; tier=1 → synonym replacement; tier=2 → single-word fallback.
    """
    words = re.sub(r"[^a-zA-Z0-9 ]", " ", query).lower().split()
    synonyms: List[str] = []

    if tier == 0:
        return [query]

    if tier == 1:
        # Replace known words with their first synonym
        replaced = []
        for w in words:
            if w in _SYNONYM_MAP:
                replaced.append(_SYNONYM_MAP[w][0])
            else:
                replaced.append(w)
        alt = " ".join(replaced)
        if alt != query.lower():
            synonyms.append(alt)
        # Also try dropping adjectives (keep nouns)
        synonyms.extend([w for w in words if len(w) > 4])
        return synonyms or [" ".join(words[:2])]

    if tier >= 2:
        # Single strongest keyword
        long_words = sorted([w for w in words if len(w) > 4], key=len, reverse=True)
        return [long_words[0]] if long_words else [words[0] if words else "video"]

    return [query]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

PIXABAY_VIDEO_API = "https://pixabay.com/api/videos/"
_CACHE_DIR = Path("./data/footage_cache/pixabay")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# License text constants (exact wording required by compliance)
PIXABAY_LICENSE_NAME = "Pixabay Content License (Free for commercial and non-commercial use. No attribution required.)"
PIXABAY_LICENSE_URL = "https://pixabay.com/service/license-summary/"
PIXABAY_RESTRICTION = "Cannot be redistributed as standalone stock footage or digital wallpaper."


_SENTINEL = object()

class PixabayProvider:
    """
    Dedicated Pixabay Video API adapter.

    All HTTP calls are made server-side; the API key never touches the browser.
    """

    def __init__(self, api_key: Any = _SENTINEL, cache_dir: Optional[Path] = None):
        if api_key is _SENTINEL:
            try:
                from src.config import settings
                api_key = settings.PIXABAY_API_KEY
            except Exception:
                api_key = None
        self._api_key: Optional[str] = api_key or None
        self.cache_dir: Path = cache_dir or _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._force_mock: bool = not bool(self._api_key)

    # ------------------------------------------------------------------
    # Public search interface
    # ------------------------------------------------------------------

    def search_videos(
        self,
        query: str,
        per_page: int = 20,
        page: int = 1,
        video_type: str = "film",
        order: str = "popular",
        min_width: int = 0,
        min_height: int = 0,
        aspect_ratio: str = "16:9",
        editors_choice: bool = False,
        target_duration: float = 5.0,
        used_ids: Optional[set] = None,
    ) -> List[PixabayVideoResult]:
        """
        Search Pixabay for videos matching *query*.

        Returns a ranked list of PixabayVideoResult objects, best match first.
        Falls back to synonym retries and ultimately mock mode on failure.
        """
        if self._force_mock:
            logger.info("Pixabay provider: API key not configured — returning mock results")
            return self._mock_results(query, aspect_ratio, target_duration, per_page)

        used_ids = used_ids or set()
        per_page = max(3, min(per_page, 50))

        # Clean query: strip visual style boilerplate so Pixabay search tags match cleanly
        clean_query = re.sub(
            r"\b(bright|modern|real-world footage|real world footage|footage|cinematic|style|visuals|video|4k|hd)\b",
            " ",
            query,
            flags=re.IGNORECASE
        )
        clean_query = re.sub(r"[^\w\s]", " ", clean_query)
        clean_query = re.sub(r"\s+", " ", clean_query).strip() or query

        for tier in range(3):
            queries = _generate_synonyms(clean_query, tier=tier)
            for q in queries:
                results = self._search_with_backoff(
                    q, per_page=per_page, page=page,
                    video_type=video_type, order=order,
                    min_width=min_width, min_height=min_height,
                    editors_choice=editors_choice,
                )
                if results is None:
                    # Unrecoverable API error — fall through to mock
                    break
                hits = [r for r in results if f"pixabay_{r.pixabay_id}" not in used_ids]
                if hits:
                    ranked = self.rank_results(hits, clean_query, target_duration, aspect_ratio)
                    logger.info(
                        f"Pixabay search '{q}' (tier {tier}): {len(hits)} usable hits "
                        f"(top score {ranked[0].relevance_score:.2f})"
                    )
                    return ranked
                logger.info(f"Pixabay search '{q}' (tier {tier}): 0 usable hits, retrying")

        # All tiers exhausted — fall back to mock
        logger.warning(f"Pixabay search exhausted all synonyms for '{query}', using mock")
        return self._mock_results(query, aspect_ratio, target_duration, per_page)

    def get_video_by_id(self, video_id: int) -> Optional[PixabayVideoResult]:
        """
        Fetch metadata for a specific Pixabay video ID.
        Returns None if not found or API is unavailable.
        """
        if self._force_mock:
            return None

        url = (
            f"{PIXABAY_VIDEO_API}?key={self._api_key}"
            f"&id={video_id}&safesearch=true"
        )
        raw = self._http_get_json(url)
        if not raw:
            return None
        hits = raw.get("hits", [])
        if not hits:
            return None
        return self._parse_hit(hits[0], query="", target_duration=5.0, aspect_ratio="16:9")

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_clip(
        self,
        result: PixabayVideoResult,
        out_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Downloads the video file and returns the local path.
        Returns None if download fails; skips download if file already exists.
        """
        asset_id = f"pixabay_{result.pixabay_id}"
        dest = out_path or (self.cache_dir / f"{asset_id}.mp4")

        if dest.exists() and dest.stat().st_size > 1024:
            return dest

        try:
            req = urllib.request.Request(
                result.download_url,
                headers={"User-Agent": "IncuBrixEngine/2.0"}
            )
            # Fast streaming with timeout and size cap (max 25MB)
            with urllib.request.urlopen(req, timeout=10) as resp, open(dest, "wb") as f:
                start_time = time.time()
                total_bytes = 0
                max_bytes = 25 * 1024 * 1024
                while True:
                    if time.time() - start_time > 15.0:
                        raise TimeoutError(f"Download took over 15s for {result.pixabay_id}")
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        break

            if dest.stat().st_size < 1024:
                logger.warning(f"Downloaded file {dest} is suspiciously small; skipping")
                dest.unlink(missing_ok=True)
                return None

            return dest
        except Exception as e:
            logger.warning(f"Pixabay download failed for {result.pixabay_id}: {e}")
            dest.unlink(missing_ok=True)
            return None

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_results(
        self,
        hits: List[PixabayVideoResult],
        query: str,
        target_duration: float,
        aspect_ratio: str,
    ) -> List[PixabayVideoResult]:
        """
        Scores each hit and returns them sorted best-first.

        Scoring weights:
          tag_overlap   × 3.0  — semantic relevance
          editors_choice × 2.0  — editorial quality signal
          resolution     × 1.5  — video quality
          duration_match × 1.0  — how close to target_duration
          popularity     × 0.5  — downloads/views signal
        """
        target_ar = self._parse_aspect_ratio(aspect_ratio)
        query_tokens = set(re.sub(r"[^a-zA-Z0-9 ]", " ", query).lower().split())

        for hit in hits:
            tag_tokens = set(re.sub(r"[^a-zA-Z0-9 ]", " ", hit.tags).lower().split())
            overlap = len(query_tokens & tag_tokens)
            tag_score = min(overlap / max(len(query_tokens), 1), 1.0)

            # Resolution score (normalise against 1920p)
            res_score = min((hit.width * hit.height) / (1920 * 1080), 1.0)

            # Duration proximity (Gaussian-ish; peak at target ±1s)
            dur_delta = abs(hit.duration - target_duration)
            dur_score = math.exp(-0.15 * dur_delta)

            # Popularity (log-scaled)
            pop_score = min(math.log1p(hit.downloads) / 15.0, 1.0)

            # Aspect ratio preference bonus
            ar_bonus = 0.0
            if target_ar and hit.height:
                hit_ar = hit.width / hit.height
                ar_diff = abs(hit_ar - target_ar)
                ar_bonus = max(0.0, 0.5 - ar_diff)

            hit.relevance_score = (
                tag_score     * 3.0
                + (1.0 if hit.editors_choice else 0.0) * 2.0
                + res_score   * 1.5
                + dur_score   * 1.0
                + pop_score   * 0.5
                + ar_bonus
            )

        hits.sort(key=lambda h: h.relevance_score, reverse=True)
        return hits

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _search_with_backoff(
        self,
        query: str,
        per_page: int,
        page: int,
        video_type: str,
        order: str,
        min_width: int,
        min_height: int,
        editors_choice: bool,
    ) -> Optional[List[PixabayVideoResult]]:
        """
        Makes the Pixabay API call with bounded exponential back-off on 429.
        Returns parsed hits or None on unrecoverable error.
        """
        params = {
            "key": self._api_key,
            "q": query,
            "video_type": video_type,
            "order": order,
            "per_page": per_page,
            "page": page,
            "safesearch": "true",
        }
        if min_width:
            params["min_width"] = min_width
        if min_height:
            params["min_height"] = min_height
        if editors_choice:
            params["editors_choice"] = "true"

        url = PIXABAY_VIDEO_API + "?" + urllib.parse.urlencode(params)

        for attempt in range(2):
            raw = self._http_get_json(url, attempt=attempt)
            if raw is None and attempt < 1:
                continue
            if raw is None:
                return None

            hits_raw = raw.get("hits", [])
            results = []
            for h in hits_raw:
                parsed = self._parse_hit(h, query=query, target_duration=5.0, aspect_ratio="16:9")
                if parsed:
                    results.append(parsed)
            return results

        return None

    def _http_get_json(self, url: str, attempt: int = 0) -> Optional[Dict[str, Any]]:
        """
        Performs an HTTP GET and returns parsed JSON.
        Handles 429 with back-off, returns None on persistent failure.
        """
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "IncuBrixEngine/2.0"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** attempt * 1.5
                logger.warning(f"Pixabay rate-limited (429); waiting {wait:.1f}s (attempt {attempt + 1}/3)")
                time.sleep(wait)
                return None
            if e.code == 400:
                logger.warning(f"Pixabay bad request (400): {e.reason}")
                return {"hits": []}
            logger.warning(f"Pixabay HTTP error {e.code}: {e.reason}")
            return None
        except Exception as e:
            logger.warning(f"Pixabay network error: {e}")
            return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_hit(
        self,
        h: Dict[str, Any],
        query: str,
        target_duration: float,
        aspect_ratio: str,
    ) -> Optional[PixabayVideoResult]:
        """
        Converts a raw API hit dict into a PixabayVideoResult.
        Selects the best available quality tier for the target aspect ratio.
        """
        try:
            v_videos = h.get("videos", {})
            # Prefer medium (1080p / 720p ~5MB) or small over huge 4K large (~80MB) for ultra-fast downloads and encoding
            tier_order = ["medium", "small", "large", "tiny"]
            chosen_tier = None
            chosen = None
            for t in tier_order:
                v = v_videos.get(t)
                if v and v.get("url"):
                    chosen_tier = t
                    chosen = v
                    break

            if not chosen:
                return None

            user = h.get("user", "Pixabay Contributor")
            user_id = h.get("user_id", 0)
            return PixabayVideoResult(
                pixabay_id=int(h["id"]),
                page_url=h.get("pageURL", ""),
                user=str(user),
                user_id=int(user_id),
                user_profile_url=f"https://pixabay.com/users/{user}-{user_id}/",
                tags=h.get("tags", ""),
                duration=int(h.get("duration", 0)),
                thumbnail_url=chosen.get("thumbnail", ""),
                download_url=chosen["url"],
                width=int(chosen.get("width", 0)),
                height=int(chosen.get("height", 0)),
                size=int(chosen.get("size", 0)),
                quality_tier=chosen_tier,
                editors_choice=bool(h.get("editor_choice", False)),
                views=int(h.get("views", 0)),
                downloads=int(h.get("downloads", 0)),
            )
        except Exception as e:
            logger.debug(f"Failed to parse Pixabay hit: {e}")
            return None

    # ------------------------------------------------------------------
    # Mock provider
    # ------------------------------------------------------------------

    def _mock_results(
        self,
        query: str,
        aspect_ratio: str,
        target_duration: float,
        count: int = 5,
    ) -> List[PixabayVideoResult]:
        """
        Returns synthetic PixabayVideoResult objects for offline dev/testing.
        Each mock result has a unique ID derived from query+index so they stay
        stable across test runs.
        """
        w, h = (1920, 1080) if aspect_ratio == "16:9" else ((1080, 1920) if aspect_ratio == "9:16" else (1080, 1080))
        results = []
        for i in range(min(count, 5)):
            seed = hashlib.md5(f"{query}_{i}".encode()).hexdigest()[:8]
            mock_id = int(seed, 16) % 999999 + 1
            score = round(4.5 - i * 0.3, 2)
            r = PixabayVideoResult(
                pixabay_id=mock_id,
                page_url=f"https://pixabay.com/videos/mock-{seed}/",
                user="MockContributor",
                user_id=12345 + i,
                user_profile_url=f"https://pixabay.com/users/mock-12345/",
                tags=query.replace(" ", ", ") + ", stock, video",
                duration=int(target_duration),
                thumbnail_url=f"/static/sample_thumb.jpg",
                download_url=f"local://mock/{seed}.mp4",
                width=w,
                height=h,
                size=5 * 1024 * 1024,
                quality_tier="large",
                editors_choice=(i == 0),
                views=10000 - i * 1000,
                downloads=5000 - i * 500,
                relevance_score=score,
                is_mock=True,
            )
            results.append(r)
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_aspect_ratio(ar: str) -> Optional[float]:
        parts = ar.split(":")
        try:
            return float(parts[0]) / float(parts[1])
        except Exception:
            return None

    def build_stock_metadata_from_result(
        self,
        result: PixabayVideoResult,
        local_path: Path,
        query: str,
        aspect_ratio: str,
        target_duration: float,
    ) -> Dict[str, Any]:
        """
        Constructs a full provenance metadata dict compatible with
        StockAssetMetadata.to_dict() and manifest_builder expectations.
        """
        asset_id = f"pixabay_{result.pixabay_id}"
        return {
            "asset_id": asset_id,
            "provider": "Pixabay Stock Video API",
            "title": result.tags or f"Stock video for {query}",
            "creator": result.user,
            "creator_url": result.user_profile_url,
            "page_url": result.page_url,
            "download_url": result.download_url,
            "license_name": PIXABAY_LICENSE_NAME,
            "license_url": PIXABAY_LICENSE_URL,
            "attribution": f"Video by {result.user} on Pixabay",
            "commercial_use_eligible": True,
            "restrictions": [PIXABAY_RESTRICTION],
            "rights_verification": "verified_provider_api",
            "has_licensing_warning": False,
            "download_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration": target_duration,
            "aspect_ratio": aspect_ratio,
            "is_mock": result.is_mock,
            "local_path": str(local_path),
            # Pixabay-specific extended fields for SOURCES.md
            "pixabay_video_id": result.pixabay_id,
            "pixabay_page_url": result.page_url,
            "contributor_user": result.user,
            "api_retrieval_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "quality_tier": result.quality_tier,
            "editors_choice": result.editors_choice,
            "relevance_score": result.relevance_score,
            "thumbnail_url": result.thumbnail_url,
            "width": result.width,
            "height": result.height,
        }


# Module-level singleton — imported by stock_service and routes
pixabay_provider = PixabayProvider()
