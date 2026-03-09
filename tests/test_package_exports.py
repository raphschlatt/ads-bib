from __future__ import annotations

from ads_bib import RunManager, apply_author_disambiguation, compute_embeddings, search_ads


def test_public_re_exports_are_importable_and_callable():
    assert RunManager is not None
    assert callable(apply_author_disambiguation)
    assert callable(search_ads)
    assert callable(compute_embeddings)
