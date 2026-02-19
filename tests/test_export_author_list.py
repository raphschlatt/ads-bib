from __future__ import annotations

import ads_bib.export as ex


def test_default_custom_format_uses_author_separator():
    assert '%ZAuthorSep:"; "' in ex.DEFAULT_CUSTOM_FORMAT
    assert '%25.25A' in ex.DEFAULT_CUSTOM_FORMAT


def test_parse_export_normalizes_author_to_list():
    raw = (
        "b1xOxTreder, H. J.; Borz, K. and Miller, A.; and 975 colleaguesxOxTitlexOx2024xOxJxOxJxOx1xOx2xOx3 xOx4 xOxAbsxOxKxOxDxOxFxOxWxOx0xOx\n"
    )
    df = ex.parse_export(raw)
    assert df.loc[0, "Author"] == ["Treder, H. J.", "Borz, K.", "Miller, A."]


def test_parse_export_strips_et_al_suffix():
    raw = (
        "b1xOxTreder, H. J.; Borz, K., et al.xOxTitlexOx2024xOxJxOxJxOx1xOx2xOx3 xOx4 xOxAbsxOxKxOxDxOxFxOxWxOx0xOx\n"
    )
    df = ex.parse_export(raw)
    assert df.loc[0, "Author"] == ["Treder, H. J.", "Borz, K."]


def test_parse_export_strips_leading_and_fragment():
    raw = (
        "b1xOxEinstein, A.; Podolsky, B.; and Rosen, N.xOxTitlexOx2024xOxJxOxJxOx1xOx2xOx3 xOx4 xOxAbsxOxKxOxDxOxFxOxWxOx0xOx\n"
    )
    df = ex.parse_export(raw)
    assert df.loc[0, "Author"] == ["Einstein, A.", "Podolsky, B.", "Rosen, N."]
