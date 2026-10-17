import pytest
from src.pipeline.scene_segmenter import scene_segmenter

def test_scene_segmenter_polysemy_cloud():
    # Context: "cloud computing"
    query, rationale = scene_segmenter._resolve_visual_query(
        text="Our servers run on cloud computing.",
        hint=None,
        avoid=[],
        style="modern"
    )
    assert "server" in query or "data center" in query
    assert "sky" not in query

def test_scene_segmenter_polysemy_bank():
    # Context: "bank account and money"
    query, rationale = scene_segmenter._resolve_visual_query(
        text="Deposit your savings in the corporate bank.",
        hint=None,
        avoid=[],
        style="modern"
    )
    assert "financial" in query or "bank" in query
    assert "river" not in query

def test_scene_segmenter_polysemy_python():
    # Context: "python programming code"
    query, rationale = scene_segmenter._resolve_visual_query(
        text="Learn python programming to automate tasks.",
        hint=None,
        avoid=[],
        style="modern"
    )
    assert "typing code" in query or "software" in query
    assert "snake" not in query

def test_scene_segmenter_avoid_filters():
    query, _ = scene_segmenter._resolve_visual_query(
        text="The team celebrated with beer.",
        hint=None,
        avoid=["people's faces", "alcohol"],
        style="modern"
    )
    assert "beer" not in query
    assert "face" not in query
