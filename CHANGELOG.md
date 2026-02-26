# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Internal release-candidate gates: packaging metadata completion, CI workflow, and install smoke checks.
- SDist smoke verification alongside wheel smoke for distribution readiness.

### Changed
- Packaging metadata in `pyproject.toml` aligned to PEP 621 (`readme`, `license`, `authors`, `classifiers`, `project.urls`).
- Dependency policy tightened: `PyYAML` added as explicit core dependency, `plotly` removed from core dependencies.
- Backlog governance clarified: `Package_ToDo.md` is the active release backlog.
- Notebook output cleanup is now enforced again for contract compliance before final release.

### Docs
- `Review_ToDo.md` archived after completion (`archive/Review_ToDo_2026-02-25_closed.md`).
- Product boundaries reaffirmed: AND stays placeholder; no BERTopic+EVoC path.
- Third-party attribution section added to `README.md`.
