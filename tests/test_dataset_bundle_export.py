from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import ads_bib.pipeline as pipeline


class _DummyTopicModel:
    def get_topic_info(self) -> pd.DataFrame:
        return pd.DataFrame({"Topic": [0, 1], "Name": ["Topic 0", "Topic 1"]})


def _topic_input_df(*, with_author_ids: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "Bibcode": "1975CMaPh..43..199H",
            "Year": 1975,
            "Author": ["Hawking, S. W."],
            "References": ["1962RSPSA.269...21B", "1962RSPSA.270..103S"],
            "tokens": ["particle", "creation", "black", "hole"],
            "Title_en": "Particle Creation by Black Holes",
            "Abstract_en": "Black hole radiation.",
        },
        {
            "Bibcode": "1975CMaPh..43..200D",
            "Year": 1975,
            "Author": ["Doe, A."],
            "References": ["1962RSPSA.269...21B", "1962RSPSA.270..103S"],
            "tokens": ["classical", "gravity", "field"],
            "Title_en": "Gravity Notes",
            "Abstract_en": "Classical field theory.",
        },
    ]
    if with_author_ids:
        rows[0]["author_uids"] = ["uid:hawking"]
        rows[0]["author_display_names"] = ["Hawking, Stephen W."]
        rows[1]["author_uids"] = ["uid:doe"]
        rows[1]["author_display_names"] = ["Doe, Alice"]
    return pd.DataFrame(rows)


def _references_df(*, with_author_ids: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "Bibcode": "1962RSPSA.269...21B",
            "Year": 1962,
            "Author": ["Bondi, H."],
            "Title_en": "Gravitational Waves in General Relativity",
        },
        {
            "Bibcode": "1962RSPSA.270..103S",
            "Year": 1962,
            "Author": ["Sachs, R. K."],
            "Title_en": "Asymptotic Symmetries in General Relativity",
        },
    ]
    if with_author_ids:
        rows[0]["author_uids"] = ["uid:bondi"]
        rows[0]["author_display_names"] = ["Bondi, Hermann"]
        rows[1]["author_uids"] = ["uid:sachs"]
        rows[1]["author_display_names"] = ["Sachs, Rainer K."]
    return pd.DataFrame(rows)


def _make_context(tmp_path, *, and_enabled: bool) -> pipeline.PipelineContext:
    config = pipeline.PipelineConfig.from_dict(
        {
            "run": {"project_root": str(tmp_path), "run_name": "bundle_export"},
            "author_disambiguation": {
                "enabled": and_enabled,
                "model_bundle": "tests://dummy-bundle" if and_enabled else None,
            },
        }
    )
    return pipeline.PipelineContext.create(config, project_root=tmp_path, load_environment=False)


def _fake_topic_dataframe(
    topic_input_df: pd.DataFrame,
    _topic_model,
    topics: np.ndarray,
    reduced_2d: np.ndarray,
    **_kwargs,
) -> pd.DataFrame:
    return topic_input_df.assign(
        topic_id=topics,
        Name=[f"Topic {int(topic)}" for topic in topics],
        embedding_2d_x=reduced_2d[:, 0],
        embedding_2d_y=reduced_2d[:, 1],
    )


def _normalized_records(df: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in df.to_dict("records"):
        normalized: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                normalized[key] = value.tolist()
            else:
                normalized[key] = value
        records.append(normalized)
    return records


def test_run_topic_dataframe_stage_writes_dataset_bundle(tmp_path, monkeypatch):
    ctx = _make_context(tmp_path, and_enabled=True)
    ctx.topic_input_df = _topic_input_df(with_author_ids=True)
    ctx.refs = _references_df(with_author_ids=True)
    ctx.topic_model = _DummyTopicModel()
    ctx.topics = np.asarray([0, 1])
    ctx.reduced_2d = np.asarray([[0.15, 0.25], [1.15, 1.25]])

    monkeypatch.setattr(pipeline, "build_topic_dataframe", _fake_topic_dataframe)

    pipeline.run_topic_dataframe_stage(ctx)

    publications_path = ctx.run.paths["data"] / "publications.parquet"
    references_path = ctx.run.paths["data"] / "references.parquet"
    manifest_path = ctx.run.paths["data"] / "dataset_manifest.json"

    assert publications_path.exists()
    assert references_path.exists()
    assert manifest_path.exists()

    publications = pd.read_parquet(publications_path)
    references = pd.read_parquet(references_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert set(publications.columns) >= {
        "Bibcode",
        "References",
        "tokens",
        "embedding_2d_x",
        "embedding_2d_y",
        "author_uids",
        "author_display_names",
    }
    assert _normalized_records(references) == _normalized_records(ctx.refs)
    assert manifest == {
        "and_enabled": True,
        "coordinate_columns": ["embedding_2d_x", "embedding_2d_y"],
        "counts": {"publications": 2, "references": 2},
        "has_author_display_names": True,
        "has_author_uids": True,
        "producer": "ads_bib",
        "producer_version": manifest["producer_version"],
        "publications_path": "publications.parquet",
        "references_path": "references.parquet",
        "run_id": ctx.run.run_id,
        "schema_version": 1,
        "source_stage": "topic_dataframe",
    }


def test_run_curate_stage_refreshes_dataset_bundle_and_manifest(tmp_path):
    ctx = _make_context(tmp_path, and_enabled=False)
    ctx.topic_df = _fake_topic_dataframe(
        _topic_input_df(with_author_ids=False),
        _DummyTopicModel(),
        np.asarray([0, 1]),
        np.asarray([[0.2, 0.3], [1.2, 1.3]]),
    )
    ctx.refs = _references_df(with_author_ids=False)

    stale_publications = pd.DataFrame([{"Bibcode": "stale-publication"}])
    stale_references = pd.DataFrame([{"Bibcode": "stale-reference"}])
    stale_publications.to_parquet(ctx.run.paths["data"] / "publications.parquet")
    stale_references.to_parquet(ctx.run.paths["data"] / "references.parquet")
    (ctx.run.paths["data"] / "dataset_manifest.json").write_text("{}", encoding="utf-8")

    pipeline.run_curate_stage(ctx)

    publications = pd.read_parquet(ctx.run.paths["data"] / "publications.parquet")
    references = pd.read_parquet(ctx.run.paths["data"] / "references.parquet")
    manifest = json.loads((ctx.run.paths["data"] / "dataset_manifest.json").read_text(encoding="utf-8"))

    assert ctx.curated_df is not None
    assert _normalized_records(publications) == _normalized_records(ctx.curated_df)
    assert _normalized_records(references) == _normalized_records(ctx.refs)
    assert manifest["source_stage"] == "curate"
    assert manifest["and_enabled"] is False
    assert manifest["has_author_uids"] is False
    assert manifest["has_author_display_names"] is False


def test_exported_dataset_bundle_loads_in_trajectories(tmp_path, monkeypatch):
    trajectories = pytest.importorskip("trajectories_of_change")
    ctx = _make_context(tmp_path, and_enabled=True)
    ctx.topic_input_df = _topic_input_df(with_author_ids=True)
    ctx.refs = _references_df(with_author_ids=True)
    ctx.topic_model = _DummyTopicModel()
    ctx.topics = np.asarray([0, 1])
    ctx.reduced_2d = np.asarray([[0.25, 0.35], [1.25, 1.35]])

    monkeypatch.setattr(pipeline, "build_topic_dataframe", _fake_topic_dataframe)

    pipeline.run_topic_dataframe_stage(ctx)

    bundle = trajectories.load_dataset_bundle(
        ctx.run.paths["data"] / "publications.parquet",
        ctx.run.paths["data"] / "references.parquet",
        manifest_path=ctx.run.paths["data"] / "dataset_manifest.json",
    )
    vocab = trajectories.VocabularyKLD(
        bundle.publications,
        "",
        target_author_uid="uid:hawking",
        window_size=1,
        skip_incomplete_slices=False,
        min_token_global_freq=1,
        min_docs_global_freq=1,
        min_tokens_target_slice=1,
        min_tokens_field_slice=1,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        min_docs_target_test=1,
        min_docs_field_test=1,
        top_k_kld_terms=2,
        legacy_name_match=False,
    )
    cocitation_df = trajectories.build_cocitation_corpus(
        bundle.publications,
        bundle.references,
        mode="works",
        target_author_uid="uid:hawking",
    )
    cocitation = trajectories.CoCitationKLD(
        cocitation_df,
        "",
        target_author_uid="uid:hawking",
        window_size=1,
        skip_incomplete_slices=False,
        min_token_global_freq=1,
        min_docs_global_freq=1,
        min_tokens_target_slice=1,
        min_tokens_field_slice=1,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        min_docs_target_test=1,
        min_docs_field_test=1,
        top_k_kld_terms=2,
        legacy_name_match=False,
    )
    density = trajectories.KDEDensity(
        bundle.publications,
        "",
        target_author_uid="uid:hawking",
        window_size=1,
        skip_incomplete_slices=False,
        bandwidth=1.0,
        min_docs_target_slice=1,
        min_docs_field_slice=1,
        legacy_name_match=False,
    )

    vocab_sync, _ = vocab.calculate_kld_sync()
    cocitation_sync, _ = cocitation.calculate_kld_sync()
    density_sync, _ = density.calculate_density_sync()

    assert bundle.manifest is not None
    assert bundle.manifest["source_stage"] == "topic_dataframe"
    assert not vocab_sync.empty
    assert not cocitation_sync.empty
    assert not density_sync.empty
