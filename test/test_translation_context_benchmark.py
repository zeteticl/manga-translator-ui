"""Benchmark guards for independent windows, source isolation and bad JSON."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("context_benchmark", Path(__file__).parents[1] / "scripts/benchmark_translation_context.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def fake_dataset():
    numbers = [n for segment in benchmark.REFERENCE_SEGMENTS for n in segment]
    ocr = {n: {"regions": [{"id": 1, "text": f"SOURCE_{n}", "translation": "DO_NOT_LEAK"}]} for n in numbers}
    vision = {n: {"scene": f"SCENE_{n}", "regions": [{"visual_evidence": "DO_NOT_LEAK_QUOTED_OCR"}]} for n in numbers}
    return ocr, vision


def test_description_and_dialogue_windows_are_independent_and_symmetric():
    import json
    ocr, vision = fake_dataset()
    payload = benchmark.window_payload(12, 1, 5, ocr, vision)
    assert [p["page"] for p in payload["previous_descriptions"]] == [11]
    assert [p["page"] for p in payload["upcoming_descriptions_reference_only"]] == [13]
    assert [p["page"] for p in payload["previous_dialogue"]] == [7, 8, 9, 10, 11]
    assert [p["page"] for p in payload["upcoming_dialogue_reference_only"]] == [13, 14, 15, 16, 17]
    assert "DO_NOT_LEAK" not in json.dumps(payload)
    assert set(payload["current_page"]) == {"page", "description", "dialogue"}


def test_zero_windows_only_include_the_current_page():
    payload = benchmark.window_payload(49, 0, 0, *fake_dataset())
    assert all(payload[key] == [] for key in payload if key != "current_page")
    assert payload["current_page"]["dialogue"] == [{"id": 1, "text": "SOURCE_49"}]


def test_all_registered_scored_pages_have_complete_largest_windows():
    for page in benchmark.FOCUS:
        before, after = benchmark.neighbours(page, 5)
        assert len(before) == len(after) == 5
        assert page not in before + after
    with pytest.raises(ValueError, match="Incomplete"):
        benchmark.neighbours(6, 1)


@pytest.mark.parametrize("items", [
    [{"id": 1, "translation：": "wrong key"}],
    [{"id": 1, "translation": 6}],
    [{"id": 1, "translation": " "}],
    [{"id": True, "translation": "bad boolean ID"}],
    [{"id": 2, "translation": "wrong ID"}],
    [{"id": 1, "translation": "text", "extra": "unexpected"}],
])
def test_malformed_fields_cannot_become_numeric_translations(items):
    with pytest.raises(ValueError):
        benchmark.validate_translations({"translations": items}, 1)


def test_id_order_is_canonical_and_duplicates_are_rejected():
    items = [{"id": 2, "translation": "second"}, {"id": 1, "translation": "first"}]
    assert benchmark.validate_translations({"translations": items}, 2) == ["first", "second"]
    with pytest.raises(ValueError):
        benchmark.validate_translations({"translations": [items[0], items[0]]}, 2)


def test_discovery_and_validation_are_disjoint():
    assert not set(benchmark.DISCOVERY) & set(benchmark.VALIDATION)
    assert set(benchmark.DISCOVERY + benchmark.VALIDATION) == set(benchmark.FOCUS)
    assert len(benchmark.variants()) == 16


def test_local_transport_usage_object_does_not_require_pydantic():
    from types import SimpleNamespace
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    assert benchmark.usage_dict(usage) == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    assert benchmark.usage_dict(None) is None
