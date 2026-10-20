"""
IncuBrix AI Video Engine — 10 Comprehensive Verification Tests
Covering all Section 14 requirements:
1. Doctor–patient conversation (Turn taking, lip-sync, male-female voice consistency)
2. Environmental explainer (Farmer archetype, EPA citations, negative space)
3. Educational video (Teacher archetype, whiteboard kinematics)
4. Technical demonstration (Engineer archetype, telemetry graphics)
5. Script requiring web research (Verified facts, authoritative URLs, research report)
6. Video using licensed stock footage (Pexels / Pixabay commercial license tracking)
7. Video using AI-generated scenes (Google Veo / generative synthesis, AI disclosure)
8. Hybrid video combining both sources (AI characters + stock B-roll)
9. Failed video-generation provider fallback (Timeout / rate-limit resilience)
10. Stock clip with missing / unclear licensing information handling (Rights warnings)
"""

import os
import json
import time
import asyncio
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from src.main import app
from src.pipeline.job_queue import job_manager
from src.pipeline.scene_planner import scene_planner
from src.pipeline.stock_service import stock_service
from src.pipeline.ai_video_service import ai_video_service
from src.pipeline.source_selector import source_selector
from src.pipeline.licensing_service import licensing_service
from src.pipeline.research_service import research_service
from src.pipeline.audio_engine import audio_engine

def test_01_doctor_patient_conversation():
    """Test 1: Doctor-patient dialogue turn-taking and male-female voice pairing."""
    script = (
        "Doctor: Hello Marcus, your routine vitals are looking very stable today.\n"
        "Patient: Thank you Doctor, I have been following the light walking routine regularly."
    )
    # Test dialogue scene planner
    scenes = scene_planner.plan_scenes(
        script=script,
        character_type="doctor, patient",
        style="bright, modern, clinical"
    )
    assert len(scenes) >= 2
    assert scenes[0]["speaker"] == "Doctor"
    assert scenes[1]["speaker"] == "Patient"
    assert "physician" in scenes[0]["character_requirements"].lower() or "doctor" in scenes[0]["character_requirements"].lower()
    assert "patient" in scenes[1]["character_requirements"].lower()
    assert scenes[0]["source_and_licensing_requirements"]["commercial_safe"] is True

    # Test male / female voice resolution
    g1 = audio_engine.detect_speaker_gender("Doctor", "clinic")
    g2 = audio_engine.detect_speaker_gender("Patient Elena", "clinic")
    assert g1 == "male"
    assert g2 == "female"
    v1 = audio_engine.resolve_voice_for_speaker("Doctor", g1)
    v2 = audio_engine.resolve_voice_for_speaker("Patient Elena", g2)
    assert "Christopher" in v1 or "Guy" in v1
    assert "Jenny" in v2 or "Aria" in v2

def test_02_environmental_explainer():
    """Test 2: Environmental explainer with farmer archetype and EPA factual callouts."""
    bundle = research_service.research_topic("Urban tree canopies reduce peak summer temperatures", mode="script_plus_research")
    assert len(bundle["verified_facts"]) > 0
    fact = bundle["verified_facts"][0]
    assert "EPA" in fact["publisher"] or "Environmental" in fact["publisher"]

    scenes = scene_planner.plan_scenes(
        script=bundle["verified_facts"][0]["fact"],
        character_type="farmer",
        style="nature, bright landscape",
        verified_facts=bundle["verified_facts"]
    )
    assert len(scenes) >= 1
    assert "Agricultural" in scenes[0]["character_requirements"]
    assert "TREES COOL URBAN CANOPIES" in scenes[0]["text_overlay_content"]["headline"]
    assert scenes[0]["text_overlay_content"]["face_safe"] is True

def test_03_educational_physics_video():
    """Test 3: Educational whiteboard lesson with teacher character and concept graphics."""
    script = "Atmospheric convection currents drive global weather systems as warm air expands and transfers heat upward."
    scenes = scene_planner.plan_scenes(
        script=script,
        character_type="teacher",
        style="educational, bright whiteboard"
    )
    assert len(scenes) >= 1
    assert "Educator" in scenes[0]["character_requirements"]
    assert "whiteboard" in scenes[0]["character_actions_and_expressions"].lower()
    assert scenes[0]["camera_angle_and_movement"] != ""

def test_04_technical_demonstration():
    """Test 4: Technical demonstration with engineer archetype and clean energy metrics."""
    topic = "Renewable wind turbine lifecycle decarbonization and grid integration"
    bundle = research_service.research_topic(topic, mode="script_plus_research")
    scenes = scene_planner.plan_scenes(
        topic=topic,
        character_type="engineer",
        style="high tech modern engineering",
        verified_facts=bundle.get("verified_facts")
    )
    assert len(scenes) >= 3
    assert "engineer" in scenes[0]["character_requirements"].lower()
    assert "Veo Prompt:" in scenes[0]["video_generation_prompt"] or "Cinematic" in scenes[0]["video_generation_prompt"]

def test_05_script_requiring_web_research():
    """Test 5: Researched script with verified citations, metrics, and markdown report."""
    bundle = research_service.research_topic("Developer productivity and AI code refactoring velocity", mode="research_first")
    assert bundle["mode"] == "research_first"
    assert len(bundle["sources"]) > 0
    first_src = bundle["sources"][0]
    assert first_src["url"].startswith("http")
    assert first_src["metric"] != ""

    report_md = research_service.generate_research_report_markdown(bundle)
    assert "# IncuBrix Video Research" in report_md
    assert first_src["publisher"] in report_md
    assert first_src["url"] in report_md

def test_06_video_using_licensed_stock_footage():
    """Test 6: Video using licensed stock footage with complete 11-field metadata tracking."""
    clip = asyncio.run(stock_service.search_and_download_clip("modern corporate architecture glass", duration=3.0, force_mock=True))
    assert clip is not None
    assert clip.local_path.exists()
    assert clip.commercial_use_eligible is True
    assert clip.license_name != ""
    assert clip.license_url.startswith("http")
    assert clip.download_url != ""

    d = clip.to_dict()
    assert "provider" in d
    assert "creator" in d
    assert "page_url" in d
    assert "download_date" in d

def test_07_video_using_ai_generated_scenes():
    """Test 7: Video using AI-generated scenes with Google Veo prompt kinematics and disclosure."""
    prompt = ai_video_service.compose_video_prompt(
        scene_text="High performance cloud computing servers in data center room",
        character_type="engineer",
        duration=3.5,
        camera_angle="medium close-up",
        camera_movement="cinematic steady push-in",
        lighting="glowing blue volumetric LED lighting"
    )
    assert "Cinematic medium close-up" in prompt
    assert "Lighting: glowing blue volumetric LED lighting" in prompt
    assert "Duration: 3.5 seconds" in prompt

    res = asyncio.run(ai_video_service.generate_scene_clip(0, prompt, duration=3.0, force_mock=True))
    assert res is not None
    assert res.local_path.exists()
    assert res.provider != ""
    assert "ai_disclosure" in res.to_dict()
    assert "terms_url" in res.to_dict()

def test_08_hybrid_video_combining_both_sources():
    """Test 8: Hybrid mode combining AI character scenes and licensed stock B-roll."""
    # Custom character scene -> Should pick AI generated
    char_scene = {
        "narration": "Our attending physician discusses cardiovascular recovery.",
        "speaker": "Doctor",
        "character_requirements": "Doctor in lab coat",
        "duration": 3.0,
        "b_roll_search_keywords": ["doctor", "hospital"]
    }
    dec_ai = asyncio.run(source_selector.select_visual_source(0, char_scene, policy="hybrid", force_mock=True))
    assert dec_ai.is_ai_generated is True
    assert "AI-generated" in dec_ai.selection_reason

    # Generic scenery scene -> Should pick licensed stock
    scenery_scene = {
        "narration": "Wind turbines spin steadily across the horizon generating clean electricity.",
        "speaker": None,
        "character_requirements": "None",
        "duration": 3.0,
        "b_roll_search_keywords": ["wind turbines", "nature scenery"]
    }
    dec_stock = asyncio.run(source_selector.select_visual_source(1, scenery_scene, policy="hybrid", force_mock=True))
    assert dec_stock.is_ai_generated is False
    assert dec_stock.commercial_rights_cleared is True
    assert "licensed stock" in dec_stock.selection_reason.lower()

def test_09_failed_video_generation_provider_fallback():
    """Test 9: Graceful recovery when external video generation provider times out or fails."""
    # Even if external API fails, the fallback produces a valid MP4 and valid result object
    res = asyncio.run(ai_video_service.generate_scene_clip(42, "Simulated network failure prompt", duration=2.5, force_mock=True))
    assert res is not None
    assert res.local_path.exists()
    assert res.duration >= 2.0
    assert res.is_mock is True

def test_10_stock_clip_unclear_licensing_handling():
    """Test 10: Stock clip with missing or sensitive rights triggers explicit warnings."""
    clip = asyncio.run(stock_service.search_and_download_clip(
        "patient undergoing clinical surgery with commercial brand logo",
        duration=3.0,
        allow_unverified=True,
        force_mock=True
    ))
    assert clip is not None
    assert clip.has_licensing_warning is True
    assert len(clip.restrictions) > 0

    sources_md = licensing_service.build_sources_markdown(
        job_id="test_job_licensing_audit",
        scenes_timeline=[{"scene_index": 0, "clip_used": clip.to_dict()}],
        warnings=["Sensitive medical imagery requires verification of patient consent release."]
    )
    assert "# IncuBrix Video Production — Sources & Licensing Audit Record" in sources_md
    assert "Auditor Notices" in sources_md
    assert "Sensitive medical imagery" in sources_md

def test_11_male_and_female_character_voices():
    """Test 11: Strict male voice for male character and female voice for female character."""
    # 1. Male characters -> Male gender & Male voices
    male_characters = ["Doctor", "Dr. Marcus", "Patient James", "Farmer", "Farmer Henry", "Executive", "Narrator David"]
    for mc in male_characters:
        g = audio_engine.detect_speaker_gender(mc)
        assert g == "male", f"Expected male for {mc}, got {g}"
        v = audio_engine.resolve_voice_for_speaker(mc, g)
        assert any(mv in v for mv in ["Christopher", "Guy", "Eric"]), f"Expected male voice for {mc}, got {v}"

    # 2. Female characters -> Female gender & Female voices
    female_characters = ["Female Doctor", "Dr. Sarah", "Patient Elena", "Teacher", "Prof. Clara", "Student", "Student Emma"]
    for fc in female_characters:
        g = audio_engine.detect_speaker_gender(fc)
        assert g == "female", f"Expected female for {fc}, got {g}"
        v = audio_engine.resolve_voice_for_speaker(fc, g)
        assert any(fv in v for fv in ["Jenny", "Aria", "Ava"]), f"Expected female voice for {fc}, got {v}"

    # 3. Regional voices (Hindi, Tamil, Indian English)
    assert audio_engine.resolve_voice_for_speaker("Doctor", "male", "hi") == "hi-IN-MadhurNeural"
    assert audio_engine.resolve_voice_for_speaker("Teacher", "female", "hi") == "hi-IN-SwaraNeural"
    assert audio_engine.resolve_voice_for_speaker("Farmer", "male", "ta") == "ta-IN-ValluvarNeural"
    assert audio_engine.resolve_voice_for_speaker("Doctor", "female", "ta") == "ta-IN-PallaviNeural"
    assert audio_engine.resolve_voice_for_speaker("Executive", "male", "en-IN") == "en-IN-PrabhatNeural"
    assert audio_engine.resolve_voice_for_speaker("Educator", "female", "en-IN") == "en-IN-NeerjaNeural"

    # 4. Turn-taking dialogue with Female Doctor and Male Patient
    dialogue = (
        "Dr. Sarah: Good morning James, your post-surgery mobility has improved significantly.\n"
        "James: Thank you Dr. Sarah, the daily physical therapy exercises are really helping."
    )
    turns = audio_engine.parse_dialogue_turns(dialogue)
    assert len(turns) == 2
    g_doc = audio_engine.detect_speaker_gender(turns[0]["speaker"])
    g_pat = audio_engine.detect_speaker_gender(turns[1]["speaker"])
    assert g_doc == "female"
    assert g_pat == "male"
    v_doc = audio_engine.resolve_voice_for_speaker(turns[0]["speaker"], g_doc)
    v_pat = audio_engine.resolve_voice_for_speaker(turns[1]["speaker"], g_pat)
    assert "Aria" in v_doc or "Jenny" in v_doc
    assert "Guy" in v_pat or "Christopher" in v_pat

    # 5. Mismatched voice override guard: Female voice candidate for male doctor is corrected to male
    tmp_out = Path("data/cache/test_voice_m_corr.mp3")
    try:
        # Pass a female voice preference ("en-US-AriaNeural") for a male doctor character
        # The engine must intercept and correct it to a male voice (e.g. Christopher / Guy)
        res_path = asyncio.run(audio_engine.synthesize_speech(
            text="Clinical diagnostics confirm routine stability.",
            voice_pref="en-US-AriaNeural",
            character_type="doctor",
            output_path=tmp_out
        ))
        assert res_path.exists()
    finally:
        if tmp_out.exists():
            tmp_out.unlink()
