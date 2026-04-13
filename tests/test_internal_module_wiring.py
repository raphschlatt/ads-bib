from __future__ import annotations

import ads_bib._dataset_bundle as dataset_bundle
import ads_bib._stage_state as stage_state
import ads_bib.notebook as notebook_module
import ads_bib.pipeline as pipeline
import ads_bib.topic_model.input as topic_input


def test_pipeline_uses_extracted_stage_state_helpers():
    assert pipeline.validate_stage_name is stage_state.validate_stage_name
    assert pipeline._earliest_invalidation_stage is stage_state._earliest_invalidation_stage
    assert pipeline._invalidate_context_from is stage_state._invalidate_context_from


def test_pipeline_uses_extracted_dataset_and_topic_helpers():
    assert pipeline._write_dataset_bundle is dataset_bundle.write_dataset_bundle
    assert pipeline._ensure_run_references_artifact is dataset_bundle.ensure_run_references_artifact
    assert pipeline._project_topic_input_frame is topic_input.project_topic_input_frame


def test_notebook_uses_stage_state_helpers_directly():
    assert notebook_module._earliest_invalidation_stage is stage_state._earliest_invalidation_stage
    assert notebook_module._invalidate_context_from is stage_state._invalidate_context_from
    assert notebook_module.validate_stage_name is stage_state.validate_stage_name
