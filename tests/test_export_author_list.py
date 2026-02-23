from __future__ import annotations

import logging

import requests

import ads_bib.export as ex


def _build_xox_raw(rows: list[list[str]]) -> str:
    return "".join("xOx".join(row) + "xOx\n" for row in rows)


def test_default_custom_format_uses_author_separator():
    assert "%ZMarkup:strip" in ex.DEFAULT_CUSTOM_FORMAT
    assert '%ZAuthorSep:"; "' in ex.DEFAULT_CUSTOM_FORMAT
    assert "%ZEncoding:csv" not in ex.DEFAULT_CUSTOM_FORMAT
    assert "%ZHeader:" not in ex.DEFAULT_CUSTOM_FORMAT
    assert '%25.25A' in ex.DEFAULT_CUSTOM_FORMAT
    assert "xOx" in ex.DEFAULT_CUSTOM_FORMAT
    assert "%p xOx%P xOx" in ex.DEFAULT_CUSTOM_FORMAT


def test_parse_export_normalizes_author_to_list():
    raw = _build_xox_raw([[
        "b1", "Treder, H. J.; Borz, K. and Miller, A.; and 975 colleagues", "Title", "2024",
        "J", "J", "1", "2", "3", "4", "Abs", "K", "D", "F", "W", "0",
    ]])
    df = ex.parse_export(raw)
    assert df.loc[0, "Author"] == ["Treder, H. J.", "Borz, K.", "Miller, A."]


def test_parse_export_strips_et_al_suffix():
    raw = _build_xox_raw([[
        "b1", "Treder, H. J.; Borz, K., et al.", "Title", "2024",
        "J", "J", "1", "2", "3", "4", "Abs", "K", "D", "F", "W", "0",
    ]])
    df = ex.parse_export(raw)
    assert df.loc[0, "Author"] == ["Treder, H. J.", "Borz, K."]


def test_parse_export_strips_leading_and_fragment():
    raw = _build_xox_raw([[
        "b1", "Einstein, A.; Podolsky, B.; and Rosen, N.", "Title", "2024",
        "J", "J", "1", "2", "3", "4", "Abs", "K", "D", "F", "W", "0",
    ]])
    df = ex.parse_export(raw)
    assert df.loc[0, "Author"] == ["Einstein, A.", "Podolsky, B.", "Rosen, N."]


def test_parse_export_drops_incomplete_xox_records():
    good_1 = _build_xox_raw([[
        "b1", "Treder, H. J.", "Title", "2024",
        "J", "J", "5", "36", "757", "763", "Abs", "K", "D", "F", "W", "0",
    ]])
    incomplete = "xOx".join([
        "bad", "Treder, H. J.", "Title", "2024", "J", "J", "5", "36", "757", "763",
        "Abs", "K", "D", "F", "W",
    ]) + "xOx\n"
    good_2 = _build_xox_raw([[
        "b2", "Einstein, A.", "Title 2", "2025",
        "J2", "J2", "1", "2", "10", "20", "Abs2", "K2", "D2", "F2", "W2", "3",
    ]])

    raw = good_1 + incomplete + good_2
    df = ex.parse_export(raw)
    assert df["Bibcode"].tolist() == ["b1", "b2"]


def test_export_bibcodes_concatenates_chunk_payloads(monkeypatch):
    class _Session:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    session = _Session()

    def _fake_create_session(token):
        assert token == "token"
        return session

    def _fake_export_chunk(
        session_obj, bibcodes_chunk, chunk_index, custom_format, max_retries, backoff_factor
    ):
        del session_obj, bibcodes_chunk, custom_format, max_retries, backoff_factor
        data = {0: "chunk0", 1: "chunk1\n"}[chunk_index]
        return chunk_index, data, None

    monkeypatch.setattr(ex, "create_session", _fake_create_session)
    monkeypatch.setattr(ex, "_export_chunk", _fake_export_chunk)
    monkeypatch.setattr(ex, "MAX_BIBCODES_PER_REQUEST", 1)

    out = ex.export_bibcodes(["b1", "b2"], "token", max_workers=2)
    assert out == "chunk0chunk1\n"
    assert session.closed is True


def test_parse_export_handles_embedded_newlines():
    raw = _build_xox_raw([[
        "b1", "A", "T", "2024", "J", "J", "1", "2", "3", "4", "Abs line 1\nAbs line 2",
        "K", "D", "F", "W", "0",
    ]])
    df = ex.parse_export(raw)
    assert df["Bibcode"].tolist() == ["b1"]
    assert "\n" not in df.loc[0, "Abstract"]


def test_parse_export_strips_bibcode_whitespace():
    raw = _build_xox_raw([[
        " b1 ", "A", "T", "2024", "J", "J", "1", "2", "3", "4", "Abs", "K", "D", "F", "W", "0",
    ]])
    df = ex.parse_export(raw)
    assert df["Bibcode"].tolist() == ["b1"]


def test_resolve_dataset_warns_when_parsed_bibcodes_are_missing(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="ads_bib.export")
    pub_raw = _build_xox_raw([[
        "b1", "A", "T", "2024", "J", "J", "1", "2", "3", "4", "Abs", "K", "D", "F", "W", "0",
    ]])
    ref_raw = _build_xox_raw([[
        "r1", "A", "T", "2024", "J", "J", "1", "2", "3", "4", "Abs", "K", "D", "F", "W", "0",
    ]])
    calls = {"n": 0}

    def _fake_export_bibcodes(*args, **kwargs):
        del args, kwargs
        calls["n"] += 1
        return pub_raw if calls["n"] == 1 else ref_raw

    monkeypatch.setattr(ex, "export_bibcodes", _fake_export_bibcodes)
    monkeypatch.setattr(ex, "clean_dataframe", lambda df: df)

    ex.resolve_dataset(
        bibcodes=["b1", "b2"],
        references=[["r1"], ["r2"]],
        esources=[[], []],
        fulltext_urls=[None, None],
        token="token",
    )
    assert "Warning: parsed publications cover 1/2 unique input bibcodes (1 missing)." in caplog.text
    assert "Warning: parsed references cover 1/2 unique input bibcodes (1 missing)." in caplog.text


def test_resolve_dataset_merges_references_and_pdf_url_after_bibcode_strip(monkeypatch):
    pub_raw = _build_xox_raw([[
        " b1", "A", "T", "2024", "J", "J", "1", "2", "3", "4", "Abs", "K", "D", "F", "W", "0",
    ]])
    ref_raw = _build_xox_raw([[
        " r1", "A", "RT", "2024", "J", "J", "1", "2", "3", "4", "Abs", "K", "D", "F", "W", "0",
    ]])
    calls = {"n": 0}

    def _fake_export_bibcodes(*args, **kwargs):
        del args, kwargs
        calls["n"] += 1
        return pub_raw if calls["n"] == 1 else ref_raw

    monkeypatch.setattr(ex, "export_bibcodes", _fake_export_bibcodes)
    monkeypatch.setattr(ex, "clean_dataframe", lambda df: df)

    pubs, refs = ex.resolve_dataset(
        bibcodes=["b1"],
        references=[["r1"]],
        esources=[[]],
        fulltext_urls=["https://example.org/pdf"],
        token="token",
    )
    assert pubs.loc[0, "Bibcode"] == "b1"
    assert pubs.loc[0, "References"] == ["r1"]
    assert pubs.loc[0, "PDF_URL"] == "https://example.org/pdf"
    assert refs.loc[0, "Bibcode"] == "r1"


def test_export_chunk_uses_retry_request_and_returns_export(monkeypatch):
    calls: dict = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"export": "ok\n"}

    def _fake_retry_request(session, method, url, **kwargs):
        del session
        calls["method"] = method
        calls["url"] = url
        calls["kwargs"] = kwargs
        return _Resp()

    monkeypatch.setattr(ex, "retry_request", _fake_retry_request)

    idx, data, err = ex._export_chunk(
        session=object(),
        bibcodes_chunk=["b1"],
        chunk_index=3,
        custom_format=ex.DEFAULT_CUSTOM_FORMAT,
    )

    assert idx == 3
    assert data == "ok\n"
    assert err is None
    assert calls["method"] == "post"
    assert calls["url"] == ex.ADS_EXPORT_URL


def test_export_chunk_returns_compact_400_error(monkeypatch):
    class _Response:
        status_code = 400

    exc = requests.exceptions.HTTPError(
        "400 Bad Request: invalid query",
        response=_Response(),
    )

    def _fake_retry_request(*args, **kwargs):
        del args, kwargs
        raise exc

    monkeypatch.setattr(ex, "retry_request", _fake_retry_request)

    idx, data, err = ex._export_chunk(
        session=object(),
        bibcodes_chunk=["b1"],
        chunk_index=1,
        custom_format=ex.DEFAULT_CUSTOM_FORMAT,
    )

    assert idx == 1
    assert data is None
    assert err == "400: invalid query"
