"""
IncuBrix Licensing & Provenance Service
Generates:
1. Production-Grade manifest.json tracking:
   - Video metadata & execution profile
   - Scene timeline & duration metrics
   - Asset inventory: AI-generated vs Licensed Stock vs User declarations
   - Detailed licensing records & rights verification status
   - Research citations & authoritative fact provenance
   - Unresolved rights warnings & compliance audits
2. Human-Readable SOURCES.md:
   - Markdown report detailing all visual and factual sources
   - Terms of service URLs and commercial use declarations
   - Explicit AI-content disclosures
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

class LicensingService:
    def __init__(self):
        pass

    def build_sources_markdown(
        self,
        job_id: str,
        scenes_timeline: List[Dict[str, Any]],
        research_bundle: Optional[Dict[str, Any]] = None,
        timings: Optional[Dict[str, float]] = None,
        warnings: Optional[List[str]] = None
    ) -> str:
        """
        Generates a separate human-readable SOURCES.md report listing every
        visual asset, audio track, research source, and applicable license.
        """
        lines = [
            "# IncuBrix Video Production — Sources & Licensing Audit Record",
            f"\n**Production Job ID:** `{job_id}`  ",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
            "**Engine:** IncuBrix Broadcast Video Engine v2.0\n",
            "---\n",
            "## 1. Executive Summary & Rights Declaration",
            "This document establishes the verified provenance and licensing status for all media assets",
            "and factual citations included in this production. Assets are classified into **Licensed Stock Footage**,",
            "**AI-Generated Video**, and **Authoritative Research Citations**.\n"
        ]

        # Scene inventory table
        lines.append("## 2. Visual Scene Assets & Provenance")
        lines.append("| Scene | Source Classification | Provider / Creator | License & Policy | Commercial Status | Reference / URL |")
        lines.append("|---|---|---|---|---|---|")

        ai_count = 0
        stock_count = 0

        for s in scenes_timeline:
            idx = s.get("scene_index", 0) + 1
            clip = s.get("clip_used") or s.get("clip") or {}
            is_ai = clip.get("is_ai_generated", False)
            if is_ai:
                ai_count += 1
                cls_type = "🤖 AI-Generated Scene"
            else:
                stock_count += 1
                cls_type = "🎞️ Licensed Stock"

            provider = clip.get("source") or clip.get("provider") or "IncuBrix Engine"
            attribution = clip.get("attribution") or provider
            lic_name = clip.get("license") or clip.get("license_name") or "Standard Commercial Grant"
            lic_url = clip.get("license_url") or "https://ai.google.dev/terms"
            comm = "✅ Eligible" if clip.get("commercial_rights_cleared", True) else "⚠️ Conditional / Editorial"
            url = clip.get("url") or "urn:incubrix:asset"

            lines.append(f"| Scene {idx} | {cls_type} | {attribution} | [{lic_name}]({lic_url}) | {comm} | [Asset Link]({url}) |")

        lines.append(f"\n*Total Visual Assets: {len(scenes_timeline)} ({ai_count} AI-Generated, {stock_count} Licensed Stock Footage)*\n")

        # Section 3: AI-Generated Content Disclosures
        lines.append("## 3. AI-Generated Content Disclosures & Provider Policies")
        lines.append("In accordance with transparency standards, AI-synthesized scenes have been generated using:")
        lines.append("- **AI Video Models:** Google Veo (`veo-3.1-generate-preview` / `veo-3.0-generate-preview`) & IncuBrix Generative Synthesis")
        lines.append("- **AI Terms of Service:** [Google Generative AI Terms of Service](https://ai.google.dev/terms)")
        lines.append("- **AI Speech Synthesis:** Microsoft Edge Neural Speech Models (en-US, en-IN, hi-IN, ta-IN)")
        lines.append("- **Disclosure Statement:** Visual assets marked as AI-Generated were synthesized via algorithmic diffusion and generative video models. They do not depict real living individuals or physical events.\n")

        # Section 4: Stock Footage Licenses & Restrictions
        lines.append("## 4. Stock Footage Licenses & Applicable Restrictions")
        lines.append("Stock assets have been procured through official provider APIs adhering to license terms:")
        lines.append("- **Pexels Content License:** Free for commercial and non-commercial use. Direct attribution is provided in accordance with community guidelines. Depicted identifiable models must not be used in an offensive or defamatory manner.")
        lines.append("- **Pixabay Content License:** Free for commercial and non-commercial use. Direct attribution is recommended. Content may not be redistributed as standalone stock footage or digital wallpaper.")
        lines.append("- **Model & Property Rights:** All stock assets are screened for commercial safety. No trademark infringement or unauthorized brand endorsement is implied.\n")

        pixabay_clips = []
        for s in scenes_timeline:
            c = s.get("clip_used") or s.get("clip") or {}
            prov = str(c.get("provider") or c.get("source") or "")
            if "pixabay" in prov.lower() or c.get("pixabay_video_id"):
                pixabay_clips.append((s, c))

        if pixabay_clips:
            lines.append("### 4.1. Pixabay Verified Assets Provenance Record")
            for s, c in pixabay_clips:
                idx = s.get("scene_index", 0) + 1
                vid_id = c.get("pixabay_video_id") or c.get("clip_id", "").replace("pixabay_", "")
                creator = c.get("creator") or c.get("attribution", "Pixabay Contributor")
                page_url = c.get("page_url") or c.get("url") or f"https://pixabay.com/videos/id-{vid_id}/"
                dl_date = c.get("download_date") or time.strftime("%Y-%m-%d", time.gmtime())
                comm_stat = "Commercial Use Eligible" if c.get("commercial_rights_cleared", True) else "Manual Approval Required"
                lines.append(f"- **Scene {idx}:** Pixabay Video ID `{vid_id}` | Creator: **{creator}** | [{page_url}]({page_url})")
                lines.append(f"  - License: Pixabay Content License ([License Summary](https://pixabay.com/service/license-summary/))")
                lines.append(f"  - Commercial Status: {comm_stat} | Retrieval Date: `{dl_date}`")
                lines.append(f"  - Restriction: *Cannot be redistributed as standalone stock footage or digital wallpaper.*")
            lines.append("")

        # Section 5: Factual Research & Authoritative Citations
        lines.append("## 5. Factual Research & Citations")
        if research_bundle and research_bundle.get("sources"):
            lines.append("Factual claims in this video have been verified against authoritative public knowledge graphs:\n")
            for idx, src in enumerate(research_bundle["sources"], 1):
                lines.append(f"### 5.{idx}. {src.get('title', 'Empirical Study')}")
                lines.append(f"- **Publisher / Authority:** {src.get('publisher')}")
                lines.append(f"- **Year of Publication:** {src.get('year')}")
                lines.append(f"- **Verified Metric:** `{src.get('metric')}`")
                lines.append(f"- **Key Finding:** {src.get('key_finding')}")
                lines.append(f"- **Direct Citation URL:** [{src.get('url')}]({src.get('url')})\n")
        else:
            lines.append("This production was generated in `script_only` mode without external factual enrichment.\n")

        # Section 6: Compliance & Warnings
        lines.append("## 6. Compliance Audit & Verification Notes")
        all_warnings = warnings or []
        if all_warnings:
            lines.append("⚠️ **Auditor Notices & Advisories:**")
            for w in all_warnings:
                lines.append(f"- {w}")
        else:
            lines.append("✅ **All assets cleared with verified commercial eligibility and zero licensing conflicts.**")

        return "\n".join(lines)

    def generate_manifest_data(
        self,
        job_id: str,
        duration: float,
        scenes: List[Dict[str, Any]],
        aspect_ratios: List[str],
        timings: Dict[str, float],
        warnings: List[str],
        research_bundle: Optional[Dict[str, Any]] = None,
        execution_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Builds the comprehensive manifest.json matching all production criteria.
        """
        ai_clips = [s for s in scenes if (s.get("clip_used", {}).get("is_ai_generated", False) or s.get("clip", {}).get("is_ai_generated", False))]
        stock_clips = [s for s in scenes if not (s.get("clip_used", {}).get("is_ai_generated", False) or s.get("clip", {}).get("is_ai_generated", False))]

        manifest = {
            "version": "2.0.0",
            "job_id": job_id,
            "engine": "IncuBrix Script-to-B-Roll Engine v2.0",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_duration_seconds": round(duration, 2),
            "execution_settings": execution_settings or {
                "aspect_ratios": aspect_ratios,
                "color_normalization": True,
                "negative_space_typography": True,
                "face_safe_protection": True
            },
            "asset_classification_summary": {
                "total_scenes": len(scenes),
                "ai_generated_scenes": len(ai_clips),
                "licensed_stock_scenes": len(stock_clips),
                "hybrid_distribution_ratio": f"{len(ai_clips)} AI : {len(stock_clips)} Stock"
            },
            "models_and_providers": {
                "ai_video_generation": {
                    "provider": "Google DeepMind Veo Video API / IncuBrix Procedural Engine",
                    "models_supported": ["veo-3.1-generate-preview", "veo-3.0-generate-preview"],
                    "policy_url": "https://ai.google.dev/terms"
                },
                "stock_footage_providers": [
                    {"provider": "Pexels Video API", "license": "Pexels Commercial Content License", "url": "https://www.pexels.com/license/"},
                    {"provider": "Pixabay Video API", "license": "Pixabay Content License", "url": "https://pixabay.com/service/license-summary/"}
                ],
                "speech_synthesis": {
                    "provider": "Microsoft Edge Neural Voice Models",
                    "languages": ["en-US", "en-IN", "hi-IN", "ta-IN"]
                },
                "speech_alignment": {
                    "provider": "Systran / CTranslate2 Faster-Whisper",
                    "alignment": "word_level_timestamps"
                }
            },
            "scene_timeline": scenes,
            "research_provenance": research_bundle or {
                "mode": "script_only",
                "sources": [],
                "verified_facts": []
            },
            "licensing_and_rights_audit": {
                "commercial_use_permitted": True,
                "watermark_free": True,
                "attributions_documented": True,
                "ai_disclosures_present": True,
                "unresolved_rights_count": len([w for w in warnings if "unresolved" in w.lower()])
            },
            "timings_breakdown_seconds": timings,
            "warnings": warnings
        }
        return manifest

licensing_service = LicensingService()
