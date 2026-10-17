import pytest
from src.pipeline.scene_segmenter import scene_segmenter
from src.pipeline.audio_engine import AudioEngineException, audio_engine

def test_hard_case_abstract_trust():
    query, rationale = scene_segmenter._resolve_visual_query(
        text="Trust is earned slowly over many years.",
        hint=None,
        avoid=[],
        style="real-world"
    )
    assert "handshake" in query or "collaboration" in query

def test_hard_case_statistics_40_percent():
    query, rationale = scene_segmenter._resolve_visual_query(
        text="Sales grew 40% in 2025 across all operations.",
        hint=None,
        avoid=[],
        style="real-world"
    )
    assert "chart" in query or "analytics" in query or "dashboard" in query

def test_hard_case_mouse_hardware():
    query, rationale = scene_segmenter._resolve_visual_query(
        text="Click the button using your desktop computer mouse.",
        hint=None,
        avoid=[],
        style="real-world"
    )
    assert "computer mouse" in query or "work desk" in query
    assert "rodent" not in query

def test_hard_case_empty_script_exception():
    with pytest.raises(AudioEngineException) as exc_info:
        import asyncio
        asyncio.run(audio_engine.synthesize_speech(""))
    assert exc_info.value.code == "EMPTY_SCRIPT_ERROR"
