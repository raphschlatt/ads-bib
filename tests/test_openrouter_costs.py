from __future__ import annotations

from types import SimpleNamespace

import ads_bib._utils.openrouter_costs as oc


def test_build_generation_endpoint_normalizes_api_base():
    assert (
        oc.build_generation_endpoint("https://openrouter.ai/api/v1")
        == "https://openrouter.ai/api/v1/generation"
    )
    assert (
        oc.build_generation_endpoint("https://openrouter.ai")
        == "https://openrouter.ai/api/v1/generation"
    )


def test_extract_usage_stats_object_and_dict():
    obj_resp = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18, cost=0.12)
    )
    dict_resp = {
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 4,
            "total_tokens": 9,
            "cost": 0.03,
        }
    }
    assert oc.extract_usage_stats(obj_resp) == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "cost": 0.12,
    }
    assert oc.extract_usage_stats(dict_resp) == {
        "prompt_tokens": 5,
        "completion_tokens": 4,
        "total_tokens": 9,
        "cost": 0.03,
    }


def test_extract_response_cost_priority():
    response = {
        "usage": {"cost": 0.1},
        "_hidden_params": {"response_cost": 0.2},
    }
    assert oc.extract_response_cost(kwargs={"response_cost": 0.3}, response=response) == 0.3
    assert oc.extract_response_cost(kwargs={}, response=response) == 0.2
    assert oc.extract_response_cost(response={"usage": {"cost": 0.4}}) == 0.4


def test_summarize_openrouter_cost_modes(monkeypatch):
    fetched = {"gen-a": 1.5, "gen-b": 2.5}

    def _fake_fetch_generation_cost(generation_id, api_key, **kwargs):
        return fetched.get(generation_id)

    monkeypatch.setattr(oc, "fetch_generation_cost", _fake_fetch_generation_cost)

    records = [
        {"generation_id": "gen-a", "direct_cost": 1.0},
        {"generation_id": "gen-b", "direct_cost": None},
        {"generation_id": None, "direct_cost": 3.0},
    ]

    hybrid = oc.summarize_openrouter_costs(
        records,
        mode="hybrid",
        api_key="x",
        wait_before_fetch=0,
    )
    assert hybrid["total_cost_usd"] == 6.5
    assert hybrid["fetch_attempted_calls"] == 1
    assert hybrid["fetched_priced_calls"] == 1

    strict = oc.summarize_openrouter_costs(
        records,
        mode="strict",
        api_key="x",
        wait_before_fetch=0,
    )
    assert strict["total_cost_usd"] == 4.0
    assert strict["fetch_attempted_calls"] == 2

    fast = oc.summarize_openrouter_costs(
        records,
        mode="fast",
        api_key="x",
        wait_before_fetch=0,
    )
    assert fast["total_cost_usd"] == 4.0
    assert fast["fetch_attempted_calls"] == 0
