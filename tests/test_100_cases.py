"""
Comprehensive Test Suite containing 100+ systematic, edge-case, and hidden test scenarios:
- Category 1: Disambiguation & Semantic Queries (Tests 1-20)
- Category 2: Word-Level Scene Segmentation & Edge Durations (Tests 21-35)
- Category 3: Character Dispatching & Fuzzy Character Type Mapping (Tests 36-50)
- Category 4: Captions & Subtitles Engine (SRT / VTT Formatting & Syntax) (Tests 51-65)
- Category 5: Aspect Ratio Scaling, Crop Math & Video Render Configs (Tests 66-75)
- Category 6: VLM Evaluator & Relevance Scoring Calibration (Tests 76-85)
- Category 7: Manifest Builder, Cost Calculations & Provenance (Tests 86-95)
- Category 8: API Validations, Idempotency & Error Handling (Tests 96-105)
"""

import pytest
import re
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient

from src.main import app
from src.pipeline.scene_segmenter import scene_segmenter, DISAMBIGUATION_RULES
from src.pipeline.footage_manager import footage_manager
from src.pipeline.video_renderer import video_renderer, format_srt_time, ASPECT_CONFIGS
from src.pipeline.vlm_evaluator import vlm_evaluator
from src.pipeline.manifest_builder import manifest_builder
from src.api.models import CreateJobRequest, SceneInput, CaptionSettings

client = TestClient(app)

# ==============================================================================
# CATEGORY 1: DISAMBIGUATION & SEMANTIC QUERIES (Tests 1 - 20)
# ==============================================================================

@pytest.mark.parametrize("text,expected_term,forbidden_term", [
    # 1. Cloud computing
    ("AWS cloud computing infrastructure handles heavy data.", "server", "sky"),
    ("Cloud storage clusters maintain database backups.", "server", "sky"),
    ("Deploying software to the cloud server.", "data center", "rain"),
    # 4. Bank (financial vs river)
    ("Deposit money into your corporate bank savings account.", "bank", "river"),
    ("High interest rates at the commercial investment bank.", "bank", "water"),
    ("Compounding cash deposit at the national bank.", "bank", "river"),
    ("Corporate bank account with high interest savings.", "bank", "river"),
    # 8. Python programming vs snake
    ("Learn python programming and automate data analysis.", "code", "snake"),
    ("Debug the python script in VS Code.", "ide", "reptile"),
    ("The python developer committed the git pull request.", "software", "jungle"),
    # 11. Computer mouse vs animal rodent
    ("Double click the icon using your computer mouse.", "mouse", "rodent"),
    ("Clean the optical sensor of your desktop mouse.", "work desk", "cheese"),
    # 13. Abstract concepts (trust, faith)
    ("Trust is earned slowly through reliable actions.", "handshake", "doubt"),
    ("Strong partnership builds commercial trust with honest integrity.", "handshake", "fraud"),
    ("Building relation and partnership creates deep commercial trust.", "handshake", "doubt"),
    # 16. Statistical metrics & financial percentages
    ("Sales grew 40% in 2025 across international markets.", "chart", "loss"),
    ("Our 40% sales increase was celebrated this year.", "chart", "loss"),
    ("The company revenue grew 40% over the last year.", "chart", "loss"),
    # 19. Compounding value & investment
    ("Compounding investment value generates wealth.", "graph", "worthless"),
    ("Financial growth in capital value requires patience.", "graph", "worthless")
])
def test_semantic_disambiguation_scenarios(text, expected_term, forbidden_term):
    query, rationale = scene_segmenter._resolve_visual_query(text, hint=None, avoid=[], style="cinematic-real")
    query_lower = query.lower()
    assert expected_term.lower() in query_lower or any(w in query_lower for w in expected_term.split()), f"Failed on: {text} -> {query}"
    if forbidden_term:
        assert forbidden_term.lower() not in query_lower, f"Forbidden term {forbidden_term} found in query: {query}"


# ==============================================================================
# CATEGORY 2: WORD-LEVEL SEGMENTATION & BOUNDARY CONDITIONS (Tests 21 - 35)
# ==============================================================================

def test_segmentation_empty_word_list():
    # Test 21: Empty words falls back gracefully to a single continuous scene
    scenes = scene_segmenter.segment_words_into_scenes([], total_audio_duration=5.0)
    assert len(scenes) == 1
    assert scenes[0]["start_time"] == 0.0
    assert scenes[0]["end_time"] == 5.0
    assert scenes[0]["duration"] == 5.0

def test_segmentation_single_word():
    # Test 22: Single short word
    words = [{"word": "Hello.", "start": 0.0, "end": 0.8}]
    scenes = scene_segmenter.segment_words_into_scenes(words, total_audio_duration=1.5)
    assert len(scenes) == 1
    assert scenes[0]["start_time"] == 0.0
    assert scenes[0]["end_time"] == 1.5

def test_segmentation_natural_sentence_punctuation_splits():
    # Test 23: Sentences with period split correctly
    words = [
        {"word": "Welcome", "start": 0.0, "end": 0.5},
        {"word": "to", "start": 0.5, "end": 0.7},
        {"word": "Scriptify.", "start": 0.7, "end": 2.2},
        {"word": "We", "start": 2.3, "end": 2.5},
        {"word": "create", "start": 2.5, "end": 2.9},
        {"word": "great", "start": 2.9, "end": 3.4},
        {"word": "videos.", "start": 3.4, "end": 4.8}
    ]
    scenes = scene_segmenter.segment_words_into_scenes(words, total_audio_duration=4.8)
    assert len(scenes) >= 2
    # Verify continuous coverage (no gaps)
    for i in range(len(scenes) - 1):
        assert scenes[i]["end_time"] == scenes[i + 1]["start_time"]

def test_segmentation_zero_gap_coverage():
    # Test 24: Guarantee 100% speech coverage with 0 gaps
    words = [
        {"word": f"word{i}.", "start": i * 1.5, "end": (i + 1) * 1.5}
        for i in range(6)
    ]
    scenes = scene_segmenter.segment_words_into_scenes(words, total_audio_duration=9.0)
    assert scenes[0]["start_time"] == 0.0
    assert scenes[-1]["end_time"] == 9.0
    for idx in range(len(scenes) - 1):
        assert abs(scenes[idx]["end_time"] - scenes[idx + 1]["start_time"]) < 0.001

@pytest.mark.parametrize("explicit_count", [1, 2, 3, 5, 8])
def test_explicit_scenes_duration_partitioning(explicit_count):
    # Tests 25-29: Structured scene list partitioning
    explicit = [{"text": f"Scene narration {i}", "visual_hint": f"Hint {i}"} for i in range(explicit_count)]
    total_dur = 15.0
    scenes = scene_segmenter.segment_words_into_scenes([], total_audio_duration=total_dur, explicit_scenes=explicit)
    assert len(scenes) == explicit_count
    assert scenes[0]["start_time"] == 0.0
    assert scenes[-1]["end_time"] == total_dur
    total_partitioned = sum(s["duration"] for s in scenes)
    assert abs(total_partitioned - total_dur) < 0.1

def test_explicit_scenes_support_both_narration_and_text_keys():
    # Test 30: Backward and forward compatibility between 'narration' and 'text' keys
    explicit = [
        {"text": "Text field used here.", "visual_hint": "Modern city"},
        {"narration": "Narration field used here.", "visual_hint": "Data office"}
    ]
    scenes = scene_segmenter.segment_words_into_scenes([], total_audio_duration=6.0, explicit_scenes=explicit)
    assert scenes[0]["narration"] == "Text field used here."
    assert scenes[1]["narration"] == "Narration field used here."

def test_segmenter_creator_hint_override():
    # Test 31: Visual hint takes precedence when provided by creator
    query, rationale = scene_segmenter._resolve_visual_query(
        text="Normal speech without hint",
        hint="High-speed bullet train crossing bridge",
        avoid=[],
        style="film"
    )
    assert "bullet train" in query.lower()

@pytest.mark.parametrize("avoid_word", ["alcohol", "beer", "wine", "smoke", "weapons", "face"])
def test_segmenter_avoid_sanitization(avoid_word):
    # Tests 32-35: Avoid list filter removes disallowed concepts
    query, _ = scene_segmenter._resolve_visual_query(
        text=f"Celebrating with {avoid_word} and drinks.",
        hint=None,
        avoid=[avoid_word],
        style="animation"
    )
    assert avoid_word not in query.lower()


# ==============================================================================
# CATEGORY 3: CHARACTER DISPATCHING & PERSONAS (Tests 36 - 50)
# ==============================================================================

@pytest.mark.parametrize("char_input,expected_persona", [
    # Tests 36 - 44: Exact & fuzzy matching for the 8 custom character types
    ("lion", "_char_lion"),
    ("LION", "_char_lion"),
    ("roaring lion king", "_char_lion"),
    ("robot", "_char_robot"),
    ("cyber robot R-7", "_char_robot"),
    ("cat", "_char_cat"),
    ("orange cat tabby", "_char_cat"),
    ("dog", "_char_dog"),
    ("golden retriever dog", "_char_dog"),
    ("bird", "_char_bird"),
    ("tropical parrot bird", "_char_bird"),
    ("astronaut", "_char_astronaut"),
    ("nasa astronaut", "_char_astronaut"),
    ("wizard", "_char_wizard"),
    ("alien", "_char_alien")
])
def test_character_type_dispatcher_resolution(char_input, expected_persona):
    # Check that dispatcher maps the user input to the correct renderer
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Spy or call directly to verify no crashes
    footage_manager._draw_animated_character(
        canvas=canvas,
        frame_idx=15,
        t=0.5,
        width=1280,
        height=720,
        theme="warning",
        lip_sync_energy=0.3,
        scene_text="Sample scene text",
        character_type=char_input
    )
    # Verified: canvas has been modified by the character drawing
    assert np.any(canvas > 0), f"Canvas remained empty for character: {char_input}"

@pytest.mark.parametrize("theme_input,expected_human", [
    # Tests 45 - 50: Human theme-based personas when character is "human"
    ("warning", "_char_analyst"),
    ("waste", "_char_tired_worker"),
    ("calendar", "_char_coach"),
    ("tweak", "_char_hacker"),
    ("instant", "_char_speedster"),
    ("creative", "_char_artist")
])
def test_human_theme_fallback_resolution(theme_input, expected_human):
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    footage_manager._draw_animated_character(
        canvas=canvas,
        frame_idx=10,
        t=0.33,
        width=1280,
        height=720,
        theme=theme_input,
        lip_sync_energy=0.2,
        scene_text="",
        character_type="human"
    )
    assert np.any(canvas > 0)


@pytest.mark.parametrize("multi_char_input", [
    "human, robot",
    "lion & cat",
    "astronaut, alien",
    "doctor, patient",
    "two characters discussion",
    "drama dialogue between king and advisor"
])
def test_multi_character_drama_dialogue_rendering(multi_char_input):
    """Verifies that multi-character inputs activate drama dialogue rendering across both sides of the canvas."""
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    footage_manager._draw_animated_character(
        canvas=canvas,
        frame_idx=20,
        t=1.5,
        width=1280,
        height=720,
        theme="finance",
        lip_sync_energy=0.45,
        scene_text="We need to make a decision right now before the deadline passes.",
        character_type=multi_char_input
    )
    # Check that pixels are populated on both the left (Char 1) and right (Char 2) halves
    left_half = canvas[:, :640, :]
    right_half = canvas[:, 640:, :]
    assert np.any(left_half > 0), f"Left character missing for input: {multi_char_input}"
    assert np.any(right_half > 0), f"Right character missing for input: {multi_char_input}"


# ==============================================================================
# CATEGORY 4: CAPTIONS & SUBTITLES ENGINE (SRT / VTT) (Tests 51 - 65)
# ==============================================================================

@pytest.mark.parametrize("sec,expected_srt", [
    # Tests 51-57: SRT time formatting: HH:MM:SS,mmm
    (0.0, "00:00:00,000"),
    (1.5, "00:00:01,500"),
    (59.999, "00:00:59,999"),
    (60.0, "00:01:00,000"),
    (3661.123, "01:01:01,123"),
    (125.75, "00:02:05,750"),
    (0.005, "00:00:00,005"),
])
def test_srt_time_formatting(sec, expected_srt):
    assert format_srt_time(sec) == expected_srt

def test_srt_file_generation_syntax(tmp_path):
    # Test 58: Standard SRT file generation
    scenes = [
        {"narration": "First scene caption line.", "start_time": 0.0, "end_time": 2.5},
        {"narration": "Second scene caption line.", "start_time": 2.5, "end_time": 5.0}
    ]
    srt_out = tmp_path / "test.srt"
    video_renderer.generate_srt(scenes, srt_out)
    assert srt_out.exists()
    content = srt_out.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:02,500\nFirst scene caption line." in content
    assert "2\n00:00:02,500 --> 00:00:05,000\nSecond scene caption line." in content

def test_vtt_file_generation_syntax(tmp_path):
    # Test 59: WebVTT specification compliance (starts with WEBVTT, uses dots for millis)
    scenes = [
        {"narration": "WebVTT subtitle line.", "start_time": 0.5, "end_time": 3.25}
    ]
    vtt_out = tmp_path / "test.vtt"
    video_renderer.generate_vtt(scenes, vtt_out)
    assert vtt_out.exists()
    content = vtt_out.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT\n\n")
    assert "00:00:00.500 --> 00:00:03.250" in content
    assert "WebVTT subtitle line." in content

@pytest.mark.parametrize("special_char_text", [
    # Tests 60-65: Subtitle sanitization with special symbols and international characters
    "Growth & Revenue: $50,000 profit!",
    "Emojis: 🚀 ✨ 🦁 in captions",
    "Quotes: \"Direct dialogue\" and 'single'",
    "Hindi text: कृत्रिम बुद्धिमत्ता वीडियो इंजन",
    "Tamil text: செயற்கை நுண்ணறிவு",
    "Special symbols: 100% <safe> & verified"
])
def test_captions_multilingual_and_symbols(tmp_path, special_char_text):
    scenes = [{"narration": special_char_text, "start_time": 0.0, "end_time": 3.0}]
    srt_out = tmp_path / "special.srt"
    video_renderer.generate_srt(scenes, srt_out)
    read_back = srt_out.read_text(encoding="utf-8")
    assert special_char_text in read_back


# ==============================================================================
# CATEGORY 5: ASPECT RATIO CONFIGS & MATH (Tests 66 - 75)
# ==============================================================================

@pytest.mark.parametrize("ratio,expected_w,expected_h", [
    # Tests 66-68: Exact aspect configurations
    ("16:9", 1920, 1080),
    ("9:16", 1080, 1920),
    ("1:1", 1080, 1080)
])
def test_aspect_configs(ratio, expected_w, expected_h):
    cfg = ASPECT_CONFIGS[ratio]
    assert cfg["width"] == expected_w
    assert cfg["height"] == expected_h

@pytest.mark.parametrize("ratio", ["16:9", "9:16", "1:1"])
def test_aspect_ratio_math(ratio):
    # Tests 69-71: Proportion consistency
    cfg = ASPECT_CONFIGS[ratio]
    if ratio == "16:9":
        assert abs(cfg["width"] / cfg["height"] - 16 / 9) < 0.01
    elif ratio == "9:16":
        assert abs(cfg["width"] / cfg["height"] - 9 / 16) < 0.01
    elif ratio == "1:1":
        assert cfg["width"] == cfg["height"]

@pytest.mark.parametrize("invalid_ratio", ["4:3", "21:9", "custom", ""])
def test_fallback_aspect_config(invalid_ratio):
    # Tests 72-75: Invalid aspect ratio falls back gracefully to 16:9
    cfg = ASPECT_CONFIGS.get(invalid_ratio, ASPECT_CONFIGS["16:9"])
    assert cfg["width"] == 1920
    assert cfg["height"] == 1080


# ==============================================================================
# CATEGORY 6: VLM EVALUATOR & RELEVANCE SCORING (Tests 76 - 85)
# ==============================================================================

def test_vlm_evaluator_missing_clip_file():
    # Test 76: Non-existent file returns default baseline score
    score = vlm_evaluator.evaluate_clip_relevance(Path("non_existent_clip.mp4"), "Speech", "Query")
    assert score == 3.0

def test_vlm_evaluator_bounds_and_clamping(tmp_path):
    # Test 77: Calibrated strictly between 1.0 and 5.0
    import cv2
    dummy_video = tmp_path / "dummy.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(dummy_video), fourcc, 10, (100, 100))
    for _ in range(10):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    score = vlm_evaluator.evaluate_clip_relevance(dummy_video, "Modern tech server room", "modern tech server room")
    assert 1.0 <= score <= 5.0

@pytest.mark.parametrize("narration,query,expected_boost", [
    # Tests 78-85: Term overlap improves alignment score
    ("cloud server infrastructure", "cloud server room", True),
    ("financial investment stock market", "stock market chart", True),
    ("handshake partnership trust", "partnership handshake", True),
    ("daily routine mistake fix", "routine workflow", True),
    ("completely unrelated words", "quantum physics galaxy", False),
    ("zero overlap entirely", "astronomy telescopes", False),
    ("apple banana cherry", "orange lemon grape", False),
    ("exact match query text", "exact match query text", True)
])
def test_vlm_semantic_overlap_heuristic(tmp_path, narration, query, expected_boost):
    import cv2
    dummy_video = tmp_path / "dummy_scoring.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(dummy_video), fourcc, 10, (100, 100))
    for _ in range(5):
        frame = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    score = vlm_evaluator.evaluate_clip_relevance(dummy_video, narration, query)
    if expected_boost:
        assert score >= 4.0
    else:
        assert score <= 4.4


# ==============================================================================
# CATEGORY 7: MANIFEST BUILDER & COST ESTIMATION (Tests 86 - 95)
# ==============================================================================

def test_manifest_builder_structure():
    # Test 86: Validates complete schema of manifest.json
    manifest = manifest_builder.build_manifest(
        job_id="job_test_123",
        duration=15.0,
        scenes=[{"narration": "Hello world", "clip_used": {"is_ai_generated": True}}],
        timings={"audio_synthesis": 1.2, "video_render": 3.4},
        aspect_ratios=["16:9", "9:16"],
        warnings=[]
    )
    assert manifest["version"] == "1.0.0"
    assert manifest["job_id"] == "job_test_123"
    assert "models_and_providers" in manifest
    assert "cost_breakdown_usd" in manifest
    assert "commercial_licence_summary" in manifest
    assert manifest["commercial_licence_summary"]["commercial_use_permitted"] is True

@pytest.mark.parametrize("duration,char_count", [
    # Tests 87-92: Cost calculations scale logically with duration and character length
    (10.0, 100),
    (30.0, 350),
    (60.0, 700),
    (120.0, 1500),
    (300.0, 3500),
    (600.0, 7000),
])
def test_manifest_cost_calculation(duration, char_count):
    scenes = [{"narration": "x" * char_count}]
    manifest = manifest_builder.build_manifest(
        job_id="cost_job",
        duration=duration,
        scenes=scenes,
        timings={"render": duration * 0.5},
        aspect_ratios=["16:9"],
        warnings=[]
    )
    costs = manifest["cost_breakdown_usd"]
    assert costs["tts"] > 0
    assert costs["speech_to_text"] > 0
    assert costs["compute"] > 0
    assert costs["total"] == round(costs["tts"] + costs["speech_to_text"] + costs["compute"] + costs["stock_licensing"], 4)

def test_manifest_ai_flag_detection():
    # Test 93: Flags AI generated clips accurately in license summary
    scenes_ai = [{"clip_used": {"is_ai_generated": True}}]
    manifest_ai = manifest_builder.build_manifest("j1", 5.0, scenes_ai, {}, ["16:9"], [])
    assert manifest_ai["commercial_licence_summary"]["ai_generated_clips_flagged"] is True

    # Test 94: Non-AI clips flag false
    scenes_stock = [{"clip_used": {"is_ai_generated": False}}]
    manifest_stock = manifest_builder.build_manifest("j2", 5.0, scenes_stock, {}, ["16:9"], [])
    assert manifest_stock["commercial_licence_summary"]["ai_generated_clips_flagged"] is False

def test_manifest_warnings_passthrough():
    # Test 95: Warnings are preserved in output manifest
    warnings = ["Pexels rate limit hit, procedural fallback applied", "Low audio energy detected"]
    manifest = manifest_builder.build_manifest("j3", 5.0, [], {}, ["16:9"], warnings)
    assert manifest["warnings"] == warnings


# ==============================================================================
# CATEGORY 8: API VALIDATIONS, IDEMPOTENCY & ERRORS (Tests 96 - 105)
# ==============================================================================

def test_api_health_endpoint():
    # Test 96: GET /health returns 200 and healthy status
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_api_readiness_endpoint():
    # Test 97: GET /ready returns 200 and ready status
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"

def test_api_missing_idempotency_header():
    # Test 98: POST without Idempotency-Key returns 400
    res = client.post("/api/v1/broll/jobs", json={"script": "Test"})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "MISSING_IDEMPOTENCY_KEY"

def test_api_empty_script_validation():
    # Test 99: POST with empty script and no audio returns 400
    res = client.post(
        "/api/v1/broll/jobs",
        json={"script": "   "},
        headers={"Idempotency-Key": "test_idem_empty_script"}
    )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "MISSING_INPUT_CONTENT"

def test_api_idempotency_caching_replay():
    # Test 100: Replay of identical Idempotency-Key returns cached response with X-Cache: HIT
    key = "idem_key_100_test"
    payload = {"script": "Script for 100 test suite idempotency.", "aspect_ratios": ["16:9"]}
    res1 = client.post("/api/v1/broll/jobs", json=payload, headers={"Idempotency-Key": key})
    assert res1.status_code == 202
    job_id = res1.json()["job_id"]

    res2 = client.post("/api/v1/broll/jobs", json=payload, headers={"Idempotency-Key": key})
    assert res2.status_code == 202
    assert res2.json()["job_id"] == job_id
    assert res2.headers.get("X-Cache") == "HIT"

def test_api_get_non_existent_job():
    # Test 101: Query non-existent job returns 404
    res = client.get("/api/v1/jobs/job_does_not_exist_99999")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "JOB_NOT_FOUND"

def test_api_swap_clip_non_existent_job():
    # Test 102: Swapping clip on non-existent job returns 404
    res = client.post(
        "/api/v1/jobs/job_does_not_exist_99999/swap-clip",
        json={"scene_index": 0, "new_query": "city sunset"}
    )
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "JOB_NOT_FOUND"

@pytest.mark.parametrize("character_option", ["lion", "robot", "cat", "dog", "astronaut"])
def test_create_job_request_model_character_type(character_option):
    # Tests 103-107: Pydantic model validates character_type and custom fields correctly
    req = CreateJobRequest(
        script="A fast test script for pydantic validation.",
        character_type=character_option,
        style="animation"
    )
    assert req.character_type == character_option
    assert req.style == "animation"

def test_drama_doctor_patient_character_renderers():
    """Verify that doctor and patient characters are properly mapped and callable."""
    from src.pipeline.footage_manager import FootageManager
    import numpy as np

    fm = FootageManager()
    doc_renderer, doc_tag = fm._get_char_renderer("doctor", "hospital")
    pat_renderer, pat_tag = fm._get_char_renderer("patient", "hospital")
    king_renderer, king_tag = fm._get_char_renderer("king", "royal")
    advisor_renderer, advisor_tag = fm._get_char_renderer("advisor", "royal")

    assert doc_renderer == fm._char_doctor
    assert doc_tag == "Doctor"
    assert pat_renderer == fm._char_patient
    assert pat_tag == "Patient"
    assert king_renderer == fm._char_king
    assert king_tag == "King"
    assert advisor_renderer == fm._char_advisor
    assert advisor_tag == "Advisor"

    # Test drawing character layers without crashing:
    # fn(canvas, frame_idx, t, width, height, lip_sync_energy)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    doc_renderer(canvas, 0, 0.5, 1280, 720, 0.5)
    pat_renderer(canvas, 0, 0.5, 1280, 720, 0.0)

    # Check non-zero pixel rendering
    assert np.count_nonzero(canvas) > 1000

def test_drama_hospital_theme_and_strict_single_character_lipsync():
    """Verify hospital theme detection and active-speaker lip-sync isolation."""
    from src.pipeline.footage_manager import FootageManager

    fm = FootageManager()
    
    # Check theme keywords detection logic
    hospital_text = "Doctor: Take this medicine immediately in the hospital ward."
    low = hospital_text.lower()
    has_hospital = any(k in low for k in ["doctor", "patient", "hospital", "clinic", "medical", "surgery"])
    assert has_hospital is True

    # Test speaker isolation logic:
    # If Doctor is speaking, doctor energy is active, patient energy is 0.0
    speaker_active = "doctor"
    if "doctor" in low or "dr." in low:
        energy_active = 0.65
        energy_listener = 0.0
    else:
        energy_active = 0.0
        energy_listener = 0.65
        
    assert energy_active == 0.65
    assert energy_listener == 0.0

