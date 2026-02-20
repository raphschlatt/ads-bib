from __future__ import annotations

from pathlib import Path

from ads_bib._utils.io import load_pickle
import ads_bib.search as search


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self):
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_search_ads_paginates_and_builds_fulltext_links(monkeypatch):
    calls: list[dict] = []
    session = _FakeSession()

    payloads = iter(
        [
            {
                "response": {
                    "docs": [
                        {"bibcode": "b1", "reference": ["r1"], "esources": ["ADS_PDF", "OTHER"]},
                        {"bibcode": "b2", "reference": [], "esources": ["PUB_HTML"]},
                    ]
                },
                "nextCursorMark": "CUR2",
            },
            {
                "response": {
                    "docs": [
                        {"bibcode": "b3", "reference": ["r2", "r3"], "esources": ["PUB_PDF"]},
                    ]
                },
                "nextCursorMark": "CUR2",
            },
        ]
    )

    def _fake_create_session(token: str):
        assert token == "token"
        return session

    def _fake_retry_request(session_obj, method, url, params):
        assert session_obj is session
        assert method == "get"
        assert url == search.ADS_SEARCH_URL
        calls.append(dict(params))
        return _FakeResponse(next(payloads))

    monkeypatch.setattr(search, "create_session", _fake_create_session)
    monkeypatch.setattr(search, "retry_request", _fake_retry_request)

    bibcodes, references, esources, fulltext_urls = search.search_ads("query", "token")

    assert bibcodes == ["b1", "b2", "b3"]
    assert references == [["r1"], [], ["r2", "r3"]]
    assert esources == [["ADS_PDF", "OTHER"], ["PUB_HTML"], ["PUB_PDF"]]
    assert fulltext_urls == [
        "https://ui.adsabs.harvard.edu/link_gateway/b1/ADS_PDF",
        None,
        "https://ui.adsabs.harvard.edu/link_gateway/b3/PUB_PDF",
    ]
    assert [c["cursorMark"] for c in calls] == ["*", "CUR2"]
    assert session.closed is True


def test_save_search_results_writes_timestamped_and_latest_files(tmp_path):
    data = (["b1"], [["r1"]], [["ADS_PDF"]], ["https://example.org"])

    saved_path = search.save_search_results(data, raw_dir=tmp_path, prefix="unit")
    latest_path = tmp_path / "unit_latest.pkl"

    assert saved_path.exists()
    assert latest_path.exists()
    assert saved_path.name.startswith("unit_")
    assert load_pickle(saved_path) == data
    assert load_pickle(latest_path) == data
