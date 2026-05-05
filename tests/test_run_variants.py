from __future__ import annotations

import yaml
import pandas as pd
import pytest

from ads_bib.run_stage_artifacts import save_frame_pair, save_search_artifact
from ads_bib.run_variants import load_base_run_config, plan_run_variant


def _base_config() -> dict[str, object]:
    return {
        "run": {"run_name": "base", "start_stage": "search", "project_root": None},
        "search": {"query": "author:test", "ads_token": "token"},
        "translate": {
            "enabled": False,
            "provider": "nllb",
            "model": "old-translate",
            "fasttext_model": "data/models/lid.176.bin",
        },
        "author_disambiguation": {"enabled": True, "dataset_id": None},
        "topic_model": {
            "backend": "bertopic",
            "embedding_provider": "openrouter",
            "embedding_model": "qwen/qwen3-embedding-8b",
            "llm_provider": "openrouter",
            "llm_model": "google/gemini-3-flash-preview",
            "params_5d": {"n_neighbors": 15},
            "cluster_params": {"min_cluster_size": 20},
        },
        "visualization": {"title": "Base Map"},
        "citations": {"min_counts": {"direct": 1}},
    }


def _write_run(tmp_path, config: dict[str, object] | None = None):
    run_dir = tmp_path / "runs" / "run_20260101_010101_base"
    run_dir.mkdir(parents=True)
    (run_dir / "config_used.yaml").write_text(
        yaml.safe_dump(config or _base_config(), sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "run_summary.yaml").write_text(
        yaml.safe_dump(
            {
                "artifact_layout_version": 2,
                "run": {"run_id": "run_20260101_010101_base"},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _write_dataset_artifacts(run_dir):
    data_dir = run_dir / "data" / "dataset"
    and_dir = run_dir / "data" / "and"
    data_dir.mkdir(parents=True)
    and_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Bibcode": "p1",
                "full_text": "alpha beta",
                "tokens": [["alpha", "beta"]],
                "author_uids": [["u1"]],
                "Title_en": "Alpha",
                "Abstract_en": "Beta",
            }
        ]
    ).to_parquet(data_dir / "publications.parquet")
    pd.DataFrame([{"Bibcode": "r1", "author_uids": [["u2"]]}]).to_parquet(
        data_dir / "references.parquet"
    )
    pd.DataFrame([{"author_uid": "u1"}]).to_parquet(and_dir / "author_entities.parquet")


def _write_stage_artifacts(run_dir):
    search_dir = run_dir / "data" / "search"
    export_dir = run_dir / "data" / "export"
    translated_dir = run_dir / "data" / "translated"
    tokenized_dir = run_dir / "data" / "tokenized"
    and_dir = run_dir / "data" / "and"
    save_search_artifact(
        search_dir,
        bibcodes=["p1"],
        references=[["r1"]],
        esources=[["ADS_PDF"]],
        fulltext_urls=["https://example.test/p1.pdf"],
    )
    exported = pd.DataFrame([{"Bibcode": "p1", "Title": "Alpha", "Abstract": "Beta"}])
    exported_refs = pd.DataFrame([{"Bibcode": "r1", "Title": "Ref", "Abstract": "Gamma"}])
    save_frame_pair(export_dir, publications=exported, refs=exported_refs)
    translated = exported.assign(Title_en="Alpha", Abstract_en="Beta")
    translated_refs = exported_refs.assign(Title_en="Ref", Abstract_en="Gamma")
    save_frame_pair(translated_dir, publications=translated, refs=translated_refs)
    tokenized = translated.assign(full_text="Alpha Beta", tokens=[["Alpha", "Beta"]])
    tokenized_refs = translated_refs.copy()
    save_frame_pair(tokenized_dir, publications=tokenized, refs=tokenized_refs)
    disambiguated = tokenized.assign(author_uids=[[["u1"]]])
    disambiguated_refs = tokenized_refs.assign(author_uids=[[["u2"]]])
    save_frame_pair(and_dir, publications=disambiguated, refs=disambiguated_refs)


def test_embedding_model_change_starts_at_embeddings(tmp_path):
    run_dir = _write_run(tmp_path)
    _write_dataset_artifacts(run_dir)

    plan = plan_run_variant(
        from_run=run_dir,
        overrides={"topic_model.embedding_model": "google/gemini-embedding-001"},
    )

    assert plan.effective_start_stage == "embeddings"
    assert plan.reused_until == "author_disambiguation"
    assert plan.changed_keys == ("topic_model.embedding_model",)
    assert plan.config.author_disambiguation.dataset_id == "run_20260101_010101_base"
    assert plan.initial_state is not None
    assert plan.initial_state.publications is not None
    assert plan.initial_state.publications["Bibcode"].tolist() == ["p1"]
    assert plan.initial_state.refs is not None
    assert plan.initial_state.refs["Bibcode"].tolist() == ["r1"]
    assert plan.initial_state.author_entities is not None


def test_reduction_params_change_starts_at_reduction(tmp_path):
    run_dir = _write_run(tmp_path)
    _write_dataset_artifacts(run_dir)

    plan = plan_run_variant(
        from_run=run_dir,
        overrides={"topic_model.params_5d.n_neighbors": 30},
    )

    assert plan.effective_start_stage == "reduction"
    assert plan.initial_state is not None
    assert plan.initial_state.publications is not None
    assert plan.initial_state.publications["Bibcode"].tolist() == ["p1"]


def test_topic_backend_clusterer_and_labeler_changes_start_at_topic_fit(tmp_path):
    run_dir = _write_run(tmp_path)
    _write_dataset_artifacts(run_dir)

    for override in (
        {"topic_model.backend": "toponymy"},
        {"topic_model.cluster_params.min_cluster_size": 40},
        {"topic_model.llm_model": "openai/gpt-4.1-mini"},
        {"topic_model.toponymy_embedding_batch_size": 128},
    ):
        plan = plan_run_variant(from_run=run_dir, overrides=override)
        assert plan.effective_start_stage == "topic_fit"


def test_citation_threshold_change_starts_at_citations_and_hydrates_base_artifacts(tmp_path):
    run_dir = _write_run(tmp_path)
    data_dir = run_dir / "data" / "dataset"
    and_dir = run_dir / "data" / "and"
    data_dir.mkdir(parents=True)
    and_dir.mkdir(parents=True)
    pd.DataFrame([{"Bibcode": "p1", "topic_id": 1}]).to_parquet(data_dir / "publications.parquet")
    pd.DataFrame([{"Bibcode": "r1"}]).to_parquet(data_dir / "references.parquet")
    pd.DataFrame([{"author_uid": "u1"}]).to_parquet(and_dir / "author_entities.parquet")

    plan = plan_run_variant(
        from_run=run_dir,
        overrides={"citations.min_counts.direct": 2},
    )

    assert plan.effective_start_stage == "citations"
    assert plan.reused_until == "curate"
    assert plan.initial_state is not None
    assert plan.initial_state.curated_df is not None
    assert plan.initial_state.curated_df["Bibcode"].tolist() == ["p1"]
    assert plan.initial_state.refs is not None
    assert plan.initial_state.refs["Bibcode"].tolist() == ["r1"]
    assert plan.initial_state.author_entities is not None
    assert plan.initial_state.author_entities["author_uid"].tolist() == ["u1"]


def test_translation_model_change_starts_at_translate(tmp_path):
    run_dir = _write_run(tmp_path)
    _write_stage_artifacts(run_dir)

    plan = plan_run_variant(
        from_run=run_dir,
        overrides={"translate.model": "new-translate"},
    )

    assert plan.effective_start_stage == "translate"
    assert plan.initial_state is not None
    assert plan.initial_state.publications is not None
    assert plan.initial_state.publications["Bibcode"].tolist() == ["p1"]
    assert "Title_en" not in plan.initial_state.publications.columns


def test_explicit_export_stage_hydrates_search_results(tmp_path):
    run_dir = _write_run(tmp_path)
    _write_stage_artifacts(run_dir)

    plan = plan_run_variant(
        from_run=run_dir,
        start_stage="export",
    )

    assert plan.effective_start_stage == "export"
    assert plan.initial_state is not None
    assert plan.initial_state.bibcodes == ["p1"]
    assert plan.initial_state.references == [["r1"]]


def test_tokenize_change_hydrates_translated_artifacts(tmp_path):
    run_dir = _write_run(tmp_path)
    _write_stage_artifacts(run_dir)

    plan = plan_run_variant(
        from_run=run_dir,
        overrides={"tokenize.batch_size": 1024},
    )

    assert plan.effective_start_stage == "tokenize"
    assert plan.initial_state is not None
    assert plan.initial_state.publications is not None
    assert "Title_en" in plan.initial_state.publications.columns
    assert "tokens" not in plan.initial_state.publications.columns


def test_author_disambiguation_change_hydrates_tokenized_artifacts(tmp_path):
    run_dir = _write_run(tmp_path)
    _write_stage_artifacts(run_dir)

    plan = plan_run_variant(
        from_run=run_dir,
        overrides={"author_disambiguation.force_refresh": True},
    )

    assert plan.effective_start_stage == "author_disambiguation"
    assert plan.initial_state is not None
    assert plan.initial_state.publications is not None
    assert "tokens" in plan.initial_state.publications.columns
    assert "author_uids" not in plan.initial_state.publications.columns


def test_visualization_change_starts_at_visualize_and_hydrates_topic_dataframe(tmp_path):
    run_dir = _write_run(tmp_path)
    data_dir = run_dir / "data" / "dataset"
    data_dir.mkdir(parents=True)
    pd.DataFrame([{"Bibcode": "p1", "topic_id": 1}]).to_parquet(data_dir / "publications.parquet")
    pd.DataFrame([{"Topic": 1, "Count": 1}]).to_parquet(data_dir / "topic_info.parquet")

    plan = plan_run_variant(
        from_run=run_dir,
        overrides={"visualization.title": "New Map"},
    )

    assert plan.effective_start_stage == "visualize"
    assert plan.initial_state is not None
    assert plan.initial_state.topic_df is not None
    assert plan.initial_state.topic_df["Bibcode"].tolist() == ["p1"]
    assert plan.initial_state.topic_info is not None
    assert plan.initial_state.topic_info["Topic"].tolist() == [1]


def test_curation_change_reruns_from_topic_fit_in_v02(tmp_path):
    run_dir = _write_run(tmp_path)
    _write_dataset_artifacts(run_dir)

    plan = plan_run_variant(
        from_run=run_dir,
        overrides={"curation.clusters_to_remove": [7]},
    )

    assert plan.effective_start_stage == "topic_fit"
    assert plan.variant["recomputed_from"] == "topic_fit"


def test_explicit_from_overrides_automatic_planning(tmp_path):
    run_dir = _write_run(tmp_path)
    _write_dataset_artifacts(run_dir)

    plan = plan_run_variant(
        from_run=run_dir,
        overrides={"topic_model.embedding_model": "google/gemini-embedding-001"},
        start_stage="topic_fit",
    )

    assert plan.effective_start_stage == "topic_fit"
    assert plan.requested_start_stage == "topic_fit"


def test_redacted_secret_placeholders_are_loaded_as_none(tmp_path):
    config = _base_config()
    config["search"] = {"query": "author:test", "ads_token": "<redacted>"}
    config["translate"] = {
        "enabled": True,
        "provider": "openrouter",
        "model": "google/gemini-3-flash-preview",
        "api_key": "<redacted>",
        "fasttext_model": "data/models/lid.176.bin",
    }
    config["topic_model"] = {
        **dict(config["topic_model"]),
        "embedding_api_key": "<redacted>",
        "llm_api_key": "<redacted>",
    }
    run_dir = _write_run(tmp_path, config)

    loaded = load_base_run_config(run_dir)

    assert loaded.search.ads_token is None
    assert loaded.translate.api_key is None
    assert loaded.topic_model.embedding_api_key is None
    assert loaded.topic_model.llm_api_key is None


def test_non_v02_base_run_layout_is_rejected(tmp_path):
    run_dir = _write_run(tmp_path)
    (run_dir / "run_summary.yaml").write_text(
        yaml.safe_dump({"run": {"run_id": "run_20260101_010101_base"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Use a v0.2 run"):
        plan_run_variant(
            from_run=run_dir,
            overrides={"topic_model.embedding_model": "google/gemini-embedding-001"},
        )
