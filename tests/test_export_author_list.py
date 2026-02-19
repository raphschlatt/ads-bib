from __future__ import annotations

import ads_bib.export as ex


def test_default_custom_format_uses_author_separator():
    assert "%ZEncoding:csv" in ex.DEFAULT_CUSTOM_FORMAT
    assert "%ZMarkup:strip" in ex.DEFAULT_CUSTOM_FORMAT
    assert '%ZAuthorSep:"; "' in ex.DEFAULT_CUSTOM_FORMAT
    assert "%ZHeader:" in ex.DEFAULT_CUSTOM_FORMAT
    assert '%25.25A' in ex.DEFAULT_CUSTOM_FORMAT
    assert "xOx" not in ex.DEFAULT_CUSTOM_FORMAT


def test_parse_export_normalizes_author_to_list():
    raw = (
        "Bibcode,Author,Title,Year,Journal,Journal Abbreviation,Issue,Volume,First Page,Last Page,Abstract,Keywords,DOI,Affiliation,Category,Citation Count\n"
        'b1,"Treder, H. J.; Borz, K. and Miller, A.; and 975 colleagues",Title,2024,J,J,1,2,3,4,Abs,K,D,F,W,0\n'
    )
    df = ex.parse_export(raw)
    assert df.loc[0, "Author"] == ["Treder, H. J.", "Borz, K.", "Miller, A."]


def test_parse_export_strips_et_al_suffix():
    raw = (
        "Bibcode,Author,Title,Year,Journal,Journal Abbreviation,Issue,Volume,First Page,Last Page,Abstract,Keywords,DOI,Affiliation,Category,Citation Count\n"
        'b1,"Treder, H. J.; Borz, K., et al.",Title,2024,J,J,1,2,3,4,Abs,K,D,F,W,0\n'
    )
    df = ex.parse_export(raw)
    assert df.loc[0, "Author"] == ["Treder, H. J.", "Borz, K."]


def test_parse_export_strips_leading_and_fragment():
    raw = (
        "Bibcode,Author,Title,Year,Journal,Journal Abbreviation,Issue,Volume,First Page,Last Page,Abstract,Keywords,DOI,Affiliation,Category,Citation Count\n"
        'b1,"Einstein, A.; Podolsky, B.; and Rosen, N.",Title,2024,J,J,1,2,3,4,Abs,K,D,F,W,0\n'
    )
    df = ex.parse_export(raw)
    assert df.loc[0, "Author"] == ["Einstein, A.", "Podolsky, B.", "Rosen, N."]


def test_parse_export_handles_broken_volume_page_csv_quoting():
    raw = (
        "Bibcode,Author,Title,Year,Journal,Journal Abbreviation,Issue,Volume,First Page,Last Page,Abstract,Keywords,DOI,Affiliation,Category,Citation Count\n"
        '"b1","Treder, H. J.","Title","2024","J","J","5","36"757",763","Abs","K","D","F","W","0"\n'
    )
    df = ex.parse_export(raw)
    assert str(df.loc[0, "Volume"]) == "36"
    assert str(df.loc[0, "First Page"]) == "757"
    assert str(df.loc[0, "Last Page"]) == "763"
