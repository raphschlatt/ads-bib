from __future__ import annotations

from ads_bib import (
    PipelineConfig,
    PipelineContext,
    NotebookSession,
    RunManager,
    RunBlockedError,
    get_notebook_session,
    run,
    run_pipeline,
)


def test_public_re_exports_are_importable_and_callable():
    assert RunManager is not None
    assert PipelineConfig is not None
    assert PipelineContext is not None
    assert NotebookSession is not None
    assert RunBlockedError is not None
    assert callable(get_notebook_session)
    assert callable(run)
    assert callable(run_pipeline)
