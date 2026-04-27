from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd
import pytest

import ads_bib.author_disambiguation as and_mod
from ads_bib.author_disambiguation import apply_author_disambiguation
from ads_bib._utils.checkpoints import save_disambiguated_snapshot
from ads_bib._utils.io import load_parquet, save_parquet


def _input_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    publications = pd.DataFrame(
        [
            {
                "Bibcode": "p1",
                "Author": ["Treder, H.J.", "Borz, K."],
                "Title_en": "Relativity",
                "Abstract_en": "Study of relativity.",
                "Affiliation": ["A", "B"],
                "Year": 1970,
            },
            {
                "Bibcode": "p2",
                "Author": ["Treder, Hans Juergen"],
                "Title_en": "Cosmology",
                "Abstract_en": "Study of cosmology.",
                "Affiliation": ["A"],
                "Year": 1971,
            },
        ]
    )
    references = pd.DataFrame(
        [
            {
                "Bibcode": "r1",
                "Author": ["Treder, H.J.", "Miller, A."],
                "Title_en": "Reference title",
                "Abstract_en": "Reference abstract.",
                "Year": 1960,
            }
        ]
    )
    return publications, references


def _write_disambiguated_outputs(
    *,
    output_root: Path,
    publications: pd.DataFrame,
    references: pd.DataFrame | None = None,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)

    publications_out = publications.copy()
    publications_out["AuthorUID"] = [
        ["uid:treder", "uid:borz"],
        ["uid:treder"],
    ]
    publications_out["AuthorDisplayName"] = [
        ["Treder, Hans Juergen", "Borz, K."],
        ["Treder, Hans Juergen"],
    ]
    publications_path = output_root / "publications_disambiguated.parquet"
    save_parquet(publications_out, publications_path)

    payload: dict[str, Path] = {"publications_disambiguated_path": publications_path}
    if references is not None:
        references_out = references.copy()
        references_out["AuthorUID"] = [["uid:treder", "uid:miller"]]
        references_out["AuthorDisplayName"] = [["Treder, Hans Juergen", "Miller, A."]]
        references_path = output_root / "references_disambiguated.parquet"
        save_parquet(references_out, references_path)
        payload["references_disambiguated_path"] = references_path

    assignments = pd.DataFrame(
        [
            {
                "source_type": "publication",
                "source_row_idx": 0,
                "bibcode": "p1",
                "author_idx": 0,
                "author_raw": "Treder, H.J.",
                "author_uid": "uid:treder",
                "assignment_kind": "clustered",
                "canonical_mention_id": "p1::0",
            }
        ]
    )
    assignments_path = output_root / "source_author_assignments.parquet"
    save_parquet(assignments, assignments_path)
    payload["source_author_assignments_path"] = assignments_path

    author_entities_path = output_root / "author_entities.parquet"
    save_parquet(pd.DataFrame([{"author_uid": "uid:treder"}]), author_entities_path)
    payload["author_entities_path"] = author_entities_path

    mention_clusters_path = output_root / "mention_clusters.parquet"
    save_parquet(pd.DataFrame([{"cluster_id": "uid:treder"}]), mention_clusters_path)
    payload["mention_clusters_path"] = mention_clusters_path

    for key, filename in {
        "summary_path": "summary.json",
        "stage_metrics_path": "05_stage_metrics_infer_sources.json",
        "go_no_go_path": "05_go_no_go_infer_sources.json",
    }.items():
        path = output_root / filename
        path.write_text('{"ok": true}', encoding="utf-8")
        payload[key] = path
    return payload


def test_default_runner_uses_public_ads_and_api_without_model_bundle(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    def _fake_disambiguate_sources(**kwargs):
        calls.update(kwargs)
        return {"publications_disambiguated_path": tmp_path / "pubs.parquet"}

    def _failing_run_infer_sources(**_kwargs):
        raise AssertionError("run_infer_sources should not be used without model_bundle")

    import author_name_disambiguation

    monkeypatch.setattr(author_name_disambiguation, "disambiguate_sources", _fake_disambiguate_sources)
    monkeypatch.setattr(author_name_disambiguation, "run_infer_sources", _failing_run_infer_sources)

    runner = and_mod._load_default_and_runner()
    runner(
        publications_path=tmp_path / "publications.parquet",
        references_path=None,
        output_dir=tmp_path / "output",
        backend="local",
        runtime="auto",
        modal_gpu="l4",
        dataset_id="dataset-1",
        infer_stage="full",
        force=False,
        progress=True,
        progress_handler=None,
        model_bundle=None,
    )

    assert calls["output_dir"] == tmp_path / "output"
    assert "model_bundle" not in calls


def test_default_runner_uses_lower_level_ads_and_api_for_model_bundle(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    def _failing_disambiguate_sources(**_kwargs):
        raise AssertionError("disambiguate_sources does not accept model_bundle")

    def _fake_run_infer_sources(**kwargs):
        calls.update(kwargs)
        return {"publications_disambiguated_path": tmp_path / "pubs.parquet"}

    import author_name_disambiguation

    monkeypatch.setattr(author_name_disambiguation, "disambiguate_sources", _failing_disambiguate_sources)
    monkeypatch.setattr(author_name_disambiguation, "run_infer_sources", _fake_run_infer_sources)

    runner = and_mod._load_default_and_runner()
    runner(
        publications_path=tmp_path / "publications.parquet",
        references_path=None,
        output_dir=tmp_path / "output",
        backend="local",
        runtime="gpu",
        modal_gpu="l4",
        dataset_id="dataset-1",
        infer_stage="full",
        force=False,
        progress=True,
        progress_handler=None,
        model_bundle="bundle-dir",
    )

    assert calls["output_root"] == tmp_path / "output"
    assert calls["model_bundle"] == "bundle-dir"
    assert calls["runtime_mode"] == "gpu"


def test_apply_author_disambiguation_stages_source_inputs_and_maps_outputs(tmp_path):
    publications, references = _input_frames()

    def _fake_runner(**kwargs):
        assert kwargs["dataset_id"] == "dataset-1"
        assert kwargs["backend"] == "local"
        assert kwargs["runtime"] == "auto"
        assert kwargs["modal_gpu"] == "l4"
        assert kwargs["model_bundle"] is None
        assert kwargs["force"] is False
        assert kwargs["infer_stage"] == "full"
        assert kwargs["progress"] is True
        assert kwargs["progress_handler"] is None

        staged_publications = load_parquet(kwargs["publications_path"])
        staged_references = load_parquet(kwargs["references_path"])
        assert staged_publications["Bibcode"].tolist() == ["p1", "p2"]
        assert staged_references["Bibcode"].tolist() == ["r1"]
        assert kwargs["output_dir"] == tmp_path / "cache" / "and_bridge" / "dataset-1" / "output"

        return _write_disambiguated_outputs(
            output_root=kwargs["output_dir"],
            publications=staged_publications,
            references=staged_references,
        )

    pubs_out, refs_out = apply_author_disambiguation(
        publications,
        references,
        dataset_id="dataset-1",
        cache_dir=tmp_path / "cache",
        run_data_dir=tmp_path / "run",
        and_runner=_fake_runner,
    )

    assert pubs_out["author_uids"].tolist() == [
        ["uid:treder", "uid:borz"],
        ["uid:treder"],
    ]
    assert pubs_out["author_display_names"].tolist() == [
        ["Treder, Hans Juergen", "Borz, K."],
        ["Treder, Hans Juergen"],
    ]
    assert refs_out["author_uids"].tolist() == [["uid:treder", "uid:miller"]]
    assert "AuthorUID" not in pubs_out.columns
    assert "AuthorDisplayName" not in pubs_out.columns
    assert (tmp_path / "run" / "publications_disambiguated.parquet").exists()
    assert (tmp_path / "run" / "references_disambiguated.parquet").exists()
    assert (tmp_path / "run" / "and" / "source_author_assignments.parquet").exists()
    assert (tmp_path / "run" / "and" / "author_entities.parquet").exists()
    assert (tmp_path / "run" / "and" / "mention_clusters.parquet").exists()
    assert (tmp_path / "run" / "and" / "summary.json").exists()
    assert (tmp_path / "run" / "and" / "05_stage_metrics_infer_sources.json").exists()
    assert (tmp_path / "run" / "and" / "05_go_no_go_infer_sources.json").exists()


def test_run_and_runner_suppresses_transformers_tracer_warning():
    from torch.jit import TracerWarning

    def _runner(**kwargs):
        warnings.warn_explicit(
            "torch.tensor results are registered as constants in the trace. You can safely ignore this warning.",
            TracerWarning,
            filename="transformers/modeling_attn_mask_utils.py",
            lineno=196,
            module="transformers.modeling_attn_mask_utils",
        )
        return kwargs["value"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = and_mod._run_and_runner(_runner, value="ok")

    assert result == "ok"
    assert caught == []


def test_apply_author_disambiguation_forwards_advanced_model_bundle(tmp_path):
    publications, _references = _input_frames()

    def _fake_runner(**kwargs):
        assert kwargs["backend"] == "modal"
        assert kwargs["runtime"] == "gpu"
        assert kwargs["modal_gpu"] == "t4"
        assert kwargs["model_bundle"] == "bundle-dir"
        staged_publications = load_parquet(kwargs["publications_path"])
        return _write_disambiguated_outputs(
            output_root=kwargs["output_dir"],
            publications=staged_publications,
            references=None,
        )

    apply_author_disambiguation(
        publications,
        None,
        backend="modal",
        runtime="gpu",
        modal_gpu="t4",
        model_bundle="bundle-dir",
        dataset_id="dataset-1",
        cache_dir=tmp_path / "cache",
        and_runner=_fake_runner,
    )


def test_apply_author_disambiguation_loads_cached_outputs_without_runner(tmp_path):
    publications, references = _input_frames()
    cache_dir = tmp_path / "cache"

    def _fake_runner(**kwargs):
        staged_publications = load_parquet(kwargs["publications_path"])
        staged_references = load_parquet(kwargs["references_path"])
        return _write_disambiguated_outputs(
            output_root=kwargs["output_dir"],
            publications=staged_publications,
            references=staged_references,
        )

    first = apply_author_disambiguation(
        publications,
        references,
        dataset_id="dataset-1",
        cache_dir=cache_dir,
        and_runner=_fake_runner,
    )

    def _failing_runner(**_kwargs):
        raise AssertionError("Runner should not be called when cache is available")

    second = apply_author_disambiguation(
        publications,
        references,
        dataset_id="dataset-1",
        cache_dir=cache_dir,
        and_runner=_failing_runner,
    )

    assert first[0]["author_uids"].tolist() == second[0]["author_uids"].tolist()
    assert first[1]["author_display_names"].tolist() == second[1]["author_display_names"].tolist()


def test_apply_author_disambiguation_ignores_unmarked_passthrough_cache(tmp_path):
    publications, references = _input_frames()
    cache_dir = tmp_path / "cache"
    passthrough_pubs = publications.assign(
        author_uids=[[""], [""]],
        author_display_names=publications["Author"],
    )
    passthrough_refs = references.assign(
        author_uids=[["", ""]],
        author_display_names=references["Author"],
    )
    save_disambiguated_snapshot(passthrough_pubs, passthrough_refs, cache_dir=cache_dir)

    calls = {"count": 0}

    def _fake_runner(**kwargs):
        calls["count"] += 1
        staged_publications = load_parquet(kwargs["publications_path"])
        staged_references = load_parquet(kwargs["references_path"])
        return _write_disambiguated_outputs(
            output_root=kwargs["output_dir"],
            publications=staged_publications,
            references=staged_references,
        )

    pubs_out, _ = apply_author_disambiguation(
        publications,
        references,
        dataset_id="dataset-1",
        cache_dir=cache_dir,
        and_runner=_fake_runner,
    )

    assert calls["count"] == 1
    assert pubs_out["author_uids"].tolist() == [
        ["uid:treder", "uid:borz"],
        ["uid:treder"],
    ]
    assert (cache_dir / "author_disambiguation_cache.json").exists()


def test_apply_author_disambiguation_supports_missing_references(tmp_path):
    publications, _references = _input_frames()

    def _fake_runner(**kwargs):
        assert kwargs["references_path"] is None
        staged_publications = load_parquet(kwargs["publications_path"])
        return _write_disambiguated_outputs(
            output_root=kwargs["output_dir"],
            publications=staged_publications,
            references=None,
        )

    pubs_out, refs_out = apply_author_disambiguation(
        publications,
        None,
        dataset_id="dataset-1",
        cache_dir=tmp_path / "cache",
        and_runner=_fake_runner,
    )

    assert pubs_out["author_uids"].tolist() == [
        ["uid:treder", "uid:borz"],
        ["uid:treder"],
    ]
    assert refs_out.empty


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda df: df.drop(columns=["AuthorDisplayName"]),
            "missing required columns",
        ),
        (
            lambda df: df.iloc[:1].copy(),
            "row count must match staged input",
        ),
        (
            lambda df: df.assign(Bibcode=["p2", "p1"]),
            "Bibcode order must match staged input",
        ),
        (
            lambda df: df.assign(AuthorUID=[["uid:treder"], ["uid:treder"]]),
            "length must match Author length",
        ),
        (
            lambda df: df.assign(
                AuthorDisplayName=[
                    ["Treder, Hans Juergen", None],
                    ["Treder, Hans Juergen"],
                ]
            ),
            "contains null entries",
        ),
    ],
)
def test_apply_author_disambiguation_validates_external_output(tmp_path, mutator, message):
    publications, references = _input_frames()

    def _fake_runner(**kwargs):
        staged_publications = load_parquet(kwargs["publications_path"])
        payload = _write_disambiguated_outputs(
            output_root=kwargs["output_dir"],
            publications=staged_publications,
            references=None,
        )
        broken = mutator(load_parquet(payload["publications_disambiguated_path"]))
        save_parquet(broken, payload["publications_disambiguated_path"])
        return payload

    with pytest.raises(ValueError, match=message):
        apply_author_disambiguation(
            publications,
            references.iloc[0:0].copy(),
            dataset_id="dataset-1",
            cache_dir=tmp_path / "cache",
            and_runner=_fake_runner,
        )
