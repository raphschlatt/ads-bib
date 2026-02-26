# ADS Pipeline (`ads-bib`) - Package ToDo

Stand: 2026-02-26
Ziel: Letzter interner RC direkt vor erstem externen Release.

## Backlog-Governance

- Diese Datei ist der aktive Release-Backlog.
- Abgeschlossener Review-Backlog liegt in:
  - `archive/Review_ToDo_2026-02-25_closed.md`

## 0) Festgelegte Entscheidungen (verbindlich)

- [x] Fokus: Release-Fundament vor neuen Forschungsfeatures.
- [x] Zielstand: interner RC (kein Public Tag in dieser Welle).
- [x] AND bleibt Placeholder bis externe Abhaengigkeit stabil ist.
  - Evidenz (2026-02-25): Dokumentiert in `AGENTS.md` Architecture Notes und in `README.md`/`CLAUDE.md`.
- [x] Kein BERTopic+EVoC-Pfad (EVoC nur via `toponymy_evoc`).
  - Evidenz (2026-02-25): Dokumentiert in `AGENTS.md` Architecture Notes und in `README.md`/`CLAUDE.md`.
- [x] Lizenz: MIT.
- [x] Dependency-Policy: kein Bloat; nur echte Core-Imports.

## 1) PR1 - Backlog und Artefakt-Klarheit

- [x] `Review_ToDo.md` aus Root entfernt und archiviert.
  - Evidenz (2026-02-25): Datei verschoben nach `archive/Review_ToDo_2026-02-25_closed.md`.
- [x] Active-vs-archived Backlog in Repo-Doku klargezogen.
  - Evidenz (2026-02-25): Hinweise in `AGENTS.md`, `README.md`, `CLAUDE.md`.
- [x] Roadmap-Grenzen fuer AND/EVoC explizit gemacht.
  - Evidenz (2026-02-25): neue Architecture Notes in `AGENTS.md` (`AND stays placeholder`, `No BERTopic+EVoC path`).

## 2) PR2 - Packaging und Dependency-Fundament

- [x] PEP-621 Metadaten in `pyproject.toml` vervollstaendigt.
  - Evidenz (2026-02-25): `readme`, `license`, `authors`, `classifiers`, `project.urls` gesetzt.
- [x] `LICENSE` (MIT) angelegt.
- [x] `CHANGELOG.md` angelegt.
- [x] Dependency-Liste konsolidiert:
  - `PyYAML` als Core-Dependency aufgenommen (Runtime-Import in `run_manager.py`).
  - `plotly` aus Core entfernt (kein aktiver Runtime-Import).
- [x] Doku-Konsistenz fuer Env-Vorlage hergestellt.
  - Evidenz (2026-02-25): Doku auf direkten `.env`-Workflow konsolidiert (ohne separate `.env.example`-Datei).

## 3) PR3 - Release-Gates und RC-Verifikation

- [x] CI-Workflow mit Python-Matrix angelegt.
  - Evidenz (2026-02-25): `.github/workflows/ci.yml` mit `3.10` und `3.12`.
- [x] CI-Gates definiert:
  - `ruff check src tests scripts`
  - `pytest -q`
  - `python -m build`
  - `python -m twine check dist/*`
  - Wheel-Smokes (Core + Topic-Extras)

### Evidenzlauf (lokal in `ADS_env`)

- [x] `ruff check src tests scripts`
  - Ergebnis (2026-02-26): `All checks passed!`
- [x] `pytest -q` (Vollsuite inkl. Notebook-Contract)
  - Ergebnis (2026-02-25): `1 failed, 149 passed` (Fail: `tests/test_pipeline_notebook_contract.py::test_pipeline_notebook_is_output_clean`)
  - Befund: `pipeline.ipynb` ist lokal nicht output-clean (`execution_count = 1` in mindestens einer Code-Zelle).
  - Fix-Evidenz (2026-02-26): nach Output-Cleanup `150 passed, 4 warnings`.
- [x] `pytest -q -k "not pipeline_notebook_contract"`
  - Ergebnis (2026-02-26): `147 passed, 3 deselected, 4 warnings`.
- [x] `pytest -q tests/test_pipeline_notebook_contract.py`
  - Ergebnis (2026-02-25): `1 failed, 2 passed` (gleicher Output-Cleanliness-Blocker).
  - Fix-Evidenz (2026-02-26): `3 passed`.
- [x] `python -m build`
  - Ergebnis (2026-02-26): `Successfully built ads_bib-0.1.0.tar.gz and ads_bib-0.1.0-py3-none-any.whl`.
- [x] `python -m twine check dist/*`
  - Ergebnis (2026-02-26): Wheel + sdist `PASSED`.
  - Hinweis: dafuer war in `ADS_env` ein Upgrade auf `packaging==26.0` noetig.
- [x] Wheel-Smoke Core (frische venv, Import-Smoke)
  - Ergebnis (2026-02-26): `wheel_core 0.1.0`.
- [x] Wheel-Smoke Topic-Extras (frische venv, Import-Smoke)
  - Ergebnis (2026-02-26): `wheel_topic_ok`.
- [x] SDist-Smoke Core (frische venv, Import-Smoke)
  - Ergebnis (2026-02-26): `sdist_core 0.1.0`.

## 4) RC-Status

- [x] Notebook output-clean und Contract gruen.
  - Evidenz (2026-02-26): `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace pipeline.ipynb`, danach `pytest -q tests/test_pipeline_notebook_contract.py` = `3 passed`.
- [x] Interner RC-Stand erreicht (ohne externen Upload/Tag).
  - Evidenz (2026-02-26): `ruff`, Teilsuite, Vollsuite, Build, Twine-Check, Wheel-Smoke und SDist-Smoke alle gruen.

## 5) Nach internem RC (nicht Teil dieser Welle)

- [ ] Public Release-Sequenz (Version bump, tag `vX.Y.Z`, Release Notes finalisieren).
- [ ] TestPyPI Upload + Install-Smoke gegen TestPyPI.
  - Blocker (2026-02-26): `TWINE_CREDS_MISSING` im lokalen Umfeld.
  - Evidenz (2026-02-26): Install-Probe gegen TestPyPI liefert `ERROR: No matching distribution found for ads-bib==0.1.0`.
- [x] Security/Compliance vor externem Release vorbereitet:
  - Secrets-Check (2026-02-26): `NO_SECRET_MATCHES`.
  - Dependency-Audit (2026-02-26, ADS_env): `75 vulnerabilities in 22 packages` (env-breit, nicht package-spezifisch).
  - Dependency-Audit (2026-02-26, frische SDist-Venv): `No known vulnerabilities found` fuer installierte Pakete; `ads-bib` als lokales Paket auf PyPI noch nicht auditierbar.

## 6) Zusaetzliche Punkte vor erstem externen Release (neu)

- [x] PyPI-Namenscheck und Release-Ziel geprueft.
  - Evidenz (2026-02-26): `https://pypi.org/pypi/ads-bib/json` und `https://test.pypi.org/pypi/ads-bib/json` liefern beide `{"message":"Not Found"}`.
- [x] SDist-Install-Smoke ergaenzt (nicht nur Wheel).
  - Evidenz (2026-02-26): Installation aus `dist/ads_bib-0.1.0.tar.gz` in frischer Venv + `sdist_core 0.1.0`.
- [x] Third-party Attribution sichtbar dokumentiert.
  - Evidenz (2026-02-26): Abschnitt `Third-Party Attribution` in `README.md`.
