"""Focused checks for the opt-in chapter workflow; no model requests are made."""

import asyncio
from types import SimpleNamespace

import pytest
from PIL import Image

from manga_translator.config import Translator
from manga_translator import extreme_mode


def test_extreme_response_ids_must_match_regions():
    with pytest.raises(ValueError, match="every text region"):
        extreme_mode._validate_vision({"scene": "room", "regions": []}, 1)
    with pytest.raises(ValueError, match="region IDs"):
        extreme_mode._validate_translations(
            {"translations": [{"id": 2, "translation": "hello"}]}, 1
        )


def test_extreme_vision_request_contains_the_page_image(monkeypatch):
    captured = {}

    async def request(_backend, messages, validate):
        captured["messages"] = messages
        return validate({
            "scene": "two people",
            "regions": [{"id": 1, "speaker": "unknown", "tone": "calm", "visual_evidence": "speech bubble"}],
        })

    monkeypatch.setattr(extreme_mode, "_request_json", request)
    image = Image.new("RGB", (8, 8), "white")
    result = asyncio.run(extreme_mode._analyze_page(None, SimpleNamespace(input=image), ["hello"]))
    content = captured["messages"][0]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert '"id": 1' in content[0]["text"]
    assert result["scene"] == "two people"


def test_extreme_analyzes_all_images_before_contextual_translation(monkeypatch, tmp_path):
    events = []
    saved = []
    pages = [str(tmp_path / f"{number:02}.png") for number in range(3)]
    cli = SimpleNamespace(
        extreme_mode=True,
        extreme_lookback_pages=2,
        extreme_lookahead_pages=2,
        extreme_vision_model=None,
        export_editable_psd=False,
        save_text=False,
    )
    config = SimpleNamespace(
        cli=cli,
        translator=SimpleNamespace(translator=Translator.openai, target_lang="CHT"),
        render=SimpleNamespace(alignment="center", direction="auto"),
    )
    items = [(page, config) for page in pages]

    class Backend:
        def parse_args(self, _config):
            pass

        async def _close_current_client(self):
            pass

    class Engine:
        verbose = False
        save_quality = 90
        text_output_file = None

        def _check_cancelled(self):
            pass

        def _cleanup_context_memory(self, _ctx, keep_result=False):
            pass

        def _detach_context_result(self, _ctx):
            pass

        async def _apply_post_translation_processing(self, ctx, _config):
            return ctx.text_regions

        async def _complete_translation_pipeline(self, ctx, _config):
            return ctx

        def _save_and_cleanup_context(self, ctx, save_info, _config, _label):
            saved.append(save_info.copy())
            ctx.output_path = str(tmp_path / "output.png")
            ctx.success = True
            ctx.result = None
            return True

    async def preprocess(_engine, item):
        page = item[0]
        events.append(("ocr", page))
        region = SimpleNamespace(text=page, translation=None)
        return SimpleNamespace(text_regions=[region], result=object(), success=False), item[1]

    async def analyze(_backend, ctx, _texts):
        events.append(("vision", ctx.text_regions[0].text))
        return {"scene": "scene", "regions": [{"id": 1, "speaker": "A", "tone": "calm", "visual_evidence": "visible"}]}

    async def translate(_backend, page, previous, upcoming, _target):
        events.append(("translate", page.path, [item.path for item in previous], [item.path for item in upcoming]))
        return ["translated"]

    monkeypatch.setattr(extreme_mode, "OpenAITranslator", Backend)
    monkeypatch.setattr(extreme_mode, "OpenAIHighQualityTranslator", Backend)
    monkeypatch.setattr(extreme_mode, "_preprocess", preprocess)
    monkeypatch.setattr(extreme_mode, "_analyze_page", analyze)
    monkeypatch.setattr(extreme_mode, "_translate_page", translate)
    monkeypatch.setattr(
        extreme_mode,
        "plan_batch_inputs",
        lambda _engine, source, _save: SimpleNamespace(
            pending_items=source, merge_results=lambda results: results
        ),
    )

    results = asyncio.run(extreme_mode.run_extreme_mode(
        Engine(), items, {"output_folder": str(tmp_path / "output"), "overwrite": True, "save_to_source_dir": True}
    ))

    assert len(results) == 3
    assert [kind for kind, *_ in events[:6]] == ["ocr", "vision"] * 3
    translations = [event for event in events if event[0] == "translate"]
    assert translations[1][2:] == ([pages[0]], [pages[2]])
    assert [result.text_regions[0].translation for result in results] == ["translated"] * 3
    assert len(saved) == 3
    assert all(info["output_folder"].startswith(str(tmp_path / "output" / "extreme")) for info in saved)
    assert all(info["overwrite"] is False and info["save_to_source_dir"] is False for info in saved)


def test_extreme_rejects_mixed_mode_without_processing():
    config = SimpleNamespace(
        cli=SimpleNamespace(extreme_mode=False),
        translator=SimpleNamespace(translator=Translator.openai),
    )
    with pytest.raises(ValueError, match="mix enabled and disabled"):
        asyncio.run(extreme_mode.run_extreme_mode(None, [("page.png", config)], None))
