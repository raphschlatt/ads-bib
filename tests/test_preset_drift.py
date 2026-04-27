"""Anti-drift checks for the four official runtime presets.

Each preset must load cleanly into :class:`PipelineConfig` and match the
road contract documented in ``docs/configuration.md`` / ``docs/runtime-roads.md``.
If a preset YAML is edited without updating the expected contract below
(or vice versa), these tests fail and force the change to surface in review.
"""

from __future__ import annotations

import pytest

from ads_bib.presets import PRESET_ORDER, load_preset_config, preset_to_dict


EXPECTED_OFFICIAL_ROADS = ("openrouter", "hf_api", "local_cpu", "local_gpu")


# Per-road contract: which provider each stage uses. These mirror the
# "Runtime Roads" matrix in the public docs. Values are the exact strings
# the code accepts in the corresponding config field.
ROAD_CONTRACT: dict[str, dict[str, str]] = {
    "openrouter": {
        "translate.provider": "openrouter",
        "topic_model.embedding_provider": "openrouter",
        "topic_model.llm_provider": "openrouter",
    },
    "hf_api": {
        "translate.provider": "huggingface_api",
        "topic_model.embedding_provider": "huggingface_api",
        "topic_model.llm_provider": "huggingface_api",
    },
    "local_cpu": {
        "translate.provider": "nllb",
        "topic_model.embedding_provider": "local",
        "topic_model.llm_provider": "llama_server",
    },
    "local_gpu": {
        "translate.provider": "transformers",
        "topic_model.embedding_provider": "local",
        "topic_model.llm_provider": "local",
    },
}


def test_preset_order_matches_official_roads():
    """`PRESET_ORDER` must list exactly the four official runtime roads."""
    assert PRESET_ORDER == EXPECTED_OFFICIAL_ROADS


@pytest.mark.parametrize("name", EXPECTED_OFFICIAL_ROADS)
def test_preset_loads_into_pipeline_config(name: str):
    """Each preset parses into a valid `PipelineConfig` without errors."""
    cfg = load_preset_config(name)
    assert cfg is not None


@pytest.mark.parametrize("name", EXPECTED_OFFICIAL_ROADS)
def test_preset_matches_road_contract(name: str):
    """Each preset wires the providers the road contract promises."""
    data = preset_to_dict(name)
    for dotted, expected in ROAD_CONTRACT[name].items():
        section, key = dotted.split(".")
        actual = data.get(section, {}).get(key)
        assert actual == expected, (
            f"preset '{name}' has {dotted}={actual!r}, expected {expected!r}. "
            f"Either the preset YAML or the documented road contract drifted."
        )


@pytest.mark.parametrize("name", EXPECTED_OFFICIAL_ROADS)
def test_preset_search_query_is_empty_placeholder(name: str):
    """Presets ship with empty `search.query`; users set it via --set or edit."""
    data = preset_to_dict(name)
    assert data.get("search", {}).get("query") == "", (
        f"preset '{name}' ships with a non-empty search.query — user input "
        "must come via CLI override or an explicit edit."
    )


@pytest.mark.parametrize("name", EXPECTED_OFFICIAL_ROADS)
def test_preset_citations_export_all_four_networks(name: str):
    """Every official preset exports the same four citation networks."""
    data = preset_to_dict(name)
    assert set(data.get("citations", {}).get("metrics", [])) == {
        "direct",
        "co_citation",
        "bibliographic_coupling",
        "author_co_citation",
    }


@pytest.mark.parametrize("name", EXPECTED_OFFICIAL_ROADS)
def test_preset_author_disambiguation_off_by_default(name: str):
    """AND is an advanced integration; presets must not enable it by default."""
    data = preset_to_dict(name)
    cfg = data.get("author_disambiguation", {})
    assert cfg.get("enabled") is False
    assert cfg.get("backend") == "local"
    assert cfg.get("runtime") == "auto"
    assert cfg.get("modal_gpu") == "l4"
    assert cfg.get("model_bundle") is None
