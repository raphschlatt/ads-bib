from __future__ import annotations

import ads_bib._utils.openrouter_client as oc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _passthrough_retry_call(func, *, max_retries, delay, backoff, on_retry=None):
    """Minimal retry_call that just calls func once (no retries)."""
    del max_retries, delay, backoff, on_retry
    return func()


def _retrying_retry_call(func, *, max_retries, delay, backoff, on_retry=None):
    """retry_call that retries on failure."""
    del delay, backoff
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as exc:
            if attempt >= max_retries:
                raise
            if on_retry is not None:
                on_retry(attempt + 1, max_retries, 0.0, exc)
    raise RuntimeError("retry_call exhausted unexpectedly.")


def _make_capturing_client():
    """Return (client, captured_kwargs_dict)."""
    captured: dict = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client(), captured


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_openrouter_chat_completion_retries_and_succeeds(monkeypatch):
    calls = {"attempts": 0}

    class _Completions:
        def create(self, **kwargs):
            del kwargs
            calls["attempts"] += 1
            if calls["attempts"] == 1:
                raise RuntimeError("transient")
            return {"ok": True}

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(oc, "retry_call", _retrying_retry_call)
    response = oc.openrouter_chat_completion(
        client=_Client(),
        model="google/gemini-3-flash-preview",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=32,
        temperature=0.0,
        retry_label="unit test",
    )

    assert calls["attempts"] == 2
    assert response == {"ok": True}


def test_extra_body_always_injects_reasoning_off(monkeypatch):
    client, captured = _make_capturing_client()
    monkeypatch.setattr(oc, "retry_call", _passthrough_retry_call)

    oc.openrouter_chat_completion(
        client=client,
        model="test/model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=32,
        temperature=0.0,
    )

    assert "extra_body" in captured
    assert captured["extra_body"]["reasoning"] == {"effort": "none"}


def test_extra_body_injects_require_parameters_with_response_format(monkeypatch):
    client, captured = _make_capturing_client()
    monkeypatch.setattr(oc, "retry_call", _passthrough_retry_call)

    oc.openrouter_chat_completion(
        client=client,
        model="test/model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=32,
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    extra = captured["extra_body"]
    assert extra["reasoning"] == {"effort": "none"}
    assert extra["provider"]["require_parameters"] is True


def test_extra_body_merges_caller_entries(monkeypatch):
    client, captured = _make_capturing_client()
    monkeypatch.setattr(oc, "retry_call", _passthrough_retry_call)

    oc.openrouter_chat_completion(
        client=client,
        model="test/model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=32,
        temperature=0.0,
        extra_body={"custom_key": "custom_val", "provider": {"order": ["Together"]}},
        response_format={"type": "json_object"},
    )

    extra = captured["extra_body"]
    assert extra["reasoning"] == {"effort": "none"}
    assert extra["custom_key"] == "custom_val"
    # Caller provider entries merged with require_parameters
    assert extra["provider"]["require_parameters"] is True
    assert extra["provider"]["order"] == ["Together"]


def test_openrouter_usage_from_response_normalizes_fields(monkeypatch):
    monkeypatch.setattr(
        oc,
        "extract_usage_stats",
        lambda response: {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
    )
    monkeypatch.setattr(oc, "extract_generation_id", lambda response: "gen_123")
    monkeypatch.setattr(oc, "extract_response_cost", lambda **kwargs: 0.0035)

    out = oc.openrouter_usage_from_response(object())
    assert out["prompt_tokens"] == 11
    assert out["completion_tokens"] == 4
    assert out["total_tokens"] == 15
    assert out["call_record"]["generation_id"] == "gen_123"
    assert out["call_record"]["direct_cost"] == 0.0035
