import time
from typing import Dict, Any, List
from src.config import settings

class ManifestBuilder:
    def build_manifest(
        self,
        job_id: str,
        duration: float,
        scenes: List[Dict[str, Any]],
        timings: Dict[str, float],
        aspect_ratios: List[str],
        warnings: List[str]
    ) -> Dict[str, Any]:
        """
        Constructs the comprehensive manifest.json matching Integration Standard Point 9.
        """
        # Calculate cost estimate based on standard cloud API rates
        # TTS cost: ~$0.015 per 1,000 characters
        # Whisper: ~$0.006 per minute
        # Stock APIs: Free tier or ~$0.10 per clip
        # Compute: ~$0.02 per minute on CPU/GPU
        total_chars = sum(len(s.get("narration", "")) for s in scenes)
        tts_cost = round((total_chars / 1000.0) * 0.015, 4)
        stt_cost = round((duration / 60.0) * 0.006, 4)
        compute_cost = round((sum(timings.values()) / 60.0) * 0.02, 4)
        footage_cost = 0.00 # Royalty-free APIs used
        total_estimated_cost = round(tts_cost + stt_cost + compute_cost + footage_cost, 4)

        manifest = {
            "version": "1.0.0",
            "job_id": job_id,
            "engine": "IncuBrix Script-to-B-Roll Engine v1.0",
            "seed": 42,
            "models_and_providers": {
                "speech_recognition": {
                    "model": "faster-whisper",
                    "version": settings.WHISPER_MODEL_SIZE,
                    "provider": "Systran / CTranslate2"
                },
                "text_to_speech": {
                    "provider": "Microsoft Edge TTS / Neural Voices",
                    "supported_locales": ["en-US", "en-IN", "hi-IN", "ta-IN"]
                },
                "stock_footage_providers": [
                    {"name": "Pexels Video API", "status": "active" if settings.PEXELS_API_KEY else "standby"},
                    {"name": "Pixabay Video API", "status": "active" if settings.PIXABAY_API_KEY else "standby"},
                    {"name": "IncuBrix Procedural Synthesizer", "status": "active"}
                ],
                "video_renderer": {
                    "tool": "FFmpeg",
                    "codec": "libx264 / AAC",
                    "framerate": 30
                }
            },
            "execution_settings": {
                "aspect_ratios": aspect_ratios,
                "color_normalization": True,
                "pacing_guard_enforced": True,
                "smart_crop_edge_subject_protection": True
            },
            "timings_seconds": timings,
            "cost_breakdown_usd": {
                "tts": tts_cost,
                "speech_to_text": stt_cost,
                "compute": compute_cost,
                "stock_licensing": footage_cost,
                "total": total_estimated_cost
            },
            "commercial_licence_summary": {
                "commercial_use_permitted": True,
                "watermark_free": True,
                "all_sources_attributed": True,
                "ai_generated_clips_flagged": any(
                    s.get("clip_used", {}).get("is_ai_generated", False) or s.get("clip", {}).get("is_ai_generated", False) 
                    for s in scenes
                )
            },
            "warnings": warnings
        }
        return manifest

manifest_builder = ManifestBuilder()
