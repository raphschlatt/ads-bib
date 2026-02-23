from __future__ import annotations

import ads_bib._utils.openrouter_client as oc


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

    def _fake_retry_call(func, *, max_retries, delay, backoff, on_retry=None):
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

    monkeypatch.setattr(oc, "retry_call", _fake_retry_call)
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
