# ADS Pipeline (`ads-bib`) - Package ToDo

Stand: 2026-03-12
Ziel: erster externer Release eines notebook-first Research-Packages mit externem AND-Adapter.

## Festgelegte Entscheidungen

- Kein `uv.lock` in dieser Release-Welle; `conda activate ADS_env` bleibt der kanonische lokale Pfad.
- AI-Dateien duerfen im Repo bleiben, aber nicht in SDist/Wheel landen; lokale `.claude/`-Dateien gehoeren nicht ins Repo.
- AND bleibt ein externes Package; `ads-bib` baut nur den ADS-spezifischen Adapter, die Validierung, das Rueck-Mapping und die Downstream-Nutzung.
- Notebook output-clean ist ein Release-Freeze-Thema und kein taeglicher Development-Gate.
- `CITATION.cff` kommt rein; ein separates `AUTHORS.md` ist fuer `0.1.x` nicht erforderlich.

## 1) Sofortige Korrekturen

- [x] `filter_by_field()` auf exaktes, case-insensitive Matching fuer Stringspalten ziehen.
  - Evidenz (2026-03-09): `src/ads_bib/curate.py` nutzt `casefold()` + `isin()` statt Regex-/Substring-Matching; `conda run -n ADS_env python -m pytest -q` -> `184 passed, 1 skipped, 4 warnings`.
- [x] `ads-bib check` auf interpreter-lokale Modulaufrufe (`python -m ...`) umstellen.
  - Evidenz (2026-03-09): `src/ads_bib/cli.py` verwendet `sys.executable -m ruff` und `sys.executable -m pytest`; `PYTHONPATH=src conda run -n ADS_env python -m ads_bib.cli check` -> `All checks passed!`.
- [x] Notebook-Cleanliness aus den Daily-Gates loesen, aber den Release-Check erhalten.
  - Evidenz (2026-03-09): `tests/test_pipeline_notebook_contract.py` skippt den Output-Cleanliness-Test ohne `ADS_CHECK_NOTEBOOK_OUTPUT=1`; `conda run -n ADS_env python -m pytest -q` bleibt gruen, ohne dass `pipeline.ipynb` gecleart sein muss.
- [x] Pandas-Warnung in `export.py` fuer `select_dtypes(include="object")` entfernen.
  - Evidenz (2026-03-09): `src/ads_bib/export.py` nutzt `include=["object", "string"]`; `conda run -n ADS_env python -m pytest -q` zeigt keine entsprechende `export.py`-Warnung.

## 2) AND-Integration in diesem Repo

- [x] Source-Input-Vertrag fuer den externen Runner festziehen.
  - Pflichtspalten: `Bibcode`, `Author`, `Year`, `Title_en`/`Title`, `Abstract_en`/`Abstract`; optional `Affiliation`.
  - Evidenz (2026-03-12): `src/ads_bib/author_disambiguation.py` validiert genau diesen source-basierten Vertrag; `README.md` dokumentiert denselben Input-Scope.
- [x] `apply_author_disambiguation()` als oeffentliche API einfuehren.
  - Evidenz (2026-03-09): Implementiert in `src/ads_bib/author_disambiguation.py`, re-exportiert via `src/ads_bib/__init__.py`; `tests/test_package_exports.py` deckt den Public Export ab.
- [x] Source-basierte Runner-Schnittstelle im Repo festziehen.
  - Runner-Aufruf ueber `publications_path`/`references_path`; Rueckgabe ueber `publications_disambiguated_path` plus optional `references_disambiguated_path`.
  - Evidenz (2026-03-12): `src/ads_bib/author_disambiguation.py` staged Inputs, ruft den externen Runner source-basiert auf und laedt die disambiguated Output-Dateien wieder ein.
- [x] Rueck-Mapping nach `publications` und `references` implementieren.
  - Zielspalten: `author_uids` und `author_display_names`, positions-aligned zu `Author`.
  - Evidenz (2026-03-09): `_apply_assignments_to_frame()`-Aequivalent liegt jetzt source-basiert in `_validate_and_normalize_output()` / `apply_author_disambiguation()`; `tests/test_author_disambiguation.py` deckt Rueck-Mapping fuer Publications und References ab.
- [x] Phase-4-Checkpoints in Parquet einfuehren.
  - Dateien: `publications_disambiguated.parquet`, `references_disambiguated.parquet`, `authors.parquet`.
  - Evidenz (2026-03-09): `src/ads_bib/_utils/checkpoints.py` implementiert `save_phase4_checkpoint()`/`load_phase4_checkpoint()`; `tests/test_checkpoints.py` deckt Schreiben, Laden und Run-Snapshot ab.
- [x] Tests mit gemocktem externen Runner einfuehren.
  - Evidenz (2026-03-12): `tests/test_author_disambiguation.py` deckt Mock-Runner-Happy-Path, Cache-Reload, fehlende References und Negativfaelle fuer defekte disambiguierte Outputs ab.
- [ ] Optionalen semantischen Konsistenzcheck fuer source-basierte AND-Outputs entscheiden.
  - Status (2026-03-12): Reihenfolge, Pflichtspalten, Listenlaengen und Nullwerte sind validiert; keine zusaetzliche globale UID-/Display-Name-Konsistenzpruefung ueber mehrere Outputs hinweg.

## 3) Anforderungen an das externe AND-Package

- [ ] Generische source-basierte API dokumentieren; keine ADS-DataFrames als externe Package-API.
  - Status (2026-03-12): Erwartung ist auf `ads-bib`-Seite in `README.md` dokumentiert; die verbindliche Festschreibung im externen AND-Package bleibt offen.
- [ ] Source-mirrored Output-Vertrag des externen Packages formal beschreiben.
  - Pflichtzusatzspalten auf Repo-Seite: `AuthorUID`, `AuthorDisplayName`.
  - Status (2026-03-12): Repo-seitige Validierung ist implementiert; der formale Vertrag des externen Packages ist noch nicht festgezogen.
- [ ] `author_display_name` als generische Entity-Metadatenlogik im externen Package festhalten.
  - Heuristik gehoert dort hin, nicht in `ads-bib`.
  - Status (2026-03-12): `README.md` verortet die menschenlesbare Repraesentationslogik bewusst im externen AND-Package; die tatsaechliche Umsetzung dort bleibt offen.

## 4) Citation-Aufraeumen

- [x] `author_co_citation` auf eigene Author-Nodes umstellen.
  - Evidenz (2026-03-09): `src/ads_bib/citations.py` baut fuer `author_co_citation` dedizierte Author-Nodes via `_build_author_nodes()`; `tests/test_citations_additional.py` assertet Author-Node-Exporte.
- [x] `author_uids` bevorzugen, Fallback auf bestehende First-Author-Name-Logik behalten.
  - Evidenz (2026-03-09): `_resolve_first_author_id()` priorisiert `author_uids` und faellt sonst auf `_first_author_lastname()` zurueck; `tests/test_citations_author_list.py` deckt UID-Priorisierung ab.
- [x] Labels aus `author_entities` ziehen, wenn vorhanden.
  - Evidenz (2026-03-09): `create_author_co_citations()` und `_build_author_nodes()` ziehen Labels aus `author_entities`; `tests/test_citations_author_list.py` deckt `author_display_name`-Labels ab.
- [x] Keine gemischten Bibcode-/Author-Knoten mehr exportieren.
  - Evidenz (2026-03-09): `process_all_citations(..., metric="author_co_citation")` exportiert nur Author-Nodes fuer diese Metrik; `tests/test_citations_additional.py` assertet ausschliesslich Author-IDs in `nodes.id`.

## 5) Release-Hygiene

- [x] `.claude/settings.local.json` aus dem Repo entfernen und `.claude/` ignorieren.
  - Evidenz (2026-03-09): `.gitignore` enthaelt `.claude/`; `git ls-files .claude/settings.local.json .claude` liefert keine Treffer.
- [x] `archive/`, `AGENTS.md`, `CLAUDE.md` und `Package_ToDo.md` aus SDist/Wheel ausschliessen.
  - Evidenz (2026-03-09): `pyproject.toml` schliesst diese Pfade im SDist aus; Artefaktpruefung nach `conda run -n ADS_env python -m build` ergibt im SDist keine Treffer und im Wheel `NO_MATCHES`.
- [ ] `archive/` aus dem Default-Branch vor externem Release neu entscheiden.
  - Status (2026-03-12): Artefakt-Ausschluss ist umgesetzt; der Verbleib von `archive/` im Repo ist noch eine gesonderte Release-Entscheidung.
- [x] `.gitignore` explizit um Coverage-/HTML-/Ruff-Artefakte ergaenzen.
  - Evidenz (2026-03-09): `.gitignore` enthaelt `.ruff_cache/`, `.coverage`, `.coverage.*` und `htmlcov/`.
- [x] SDist/Wheel-Inhalte nach jedem Packaging-Change pruefen.
  - Erfolgskriterium: keine lokalen Tooling-Dateien und keine internen Backlogs im Artefakt.
  - Evidenz (2026-03-12): `python -m build` -> sdist und wheel erfolgreich gebaut; `python -m twine check dist/*` -> beide Artefakte `PASSED`; Inhaltspruefungen fuer SDist/Wheel zeigen keine Treffer fuer `archive/`, `AGENTS.md`, `CLAUDE.md`, `Package_ToDo.md`.

## 6) Metadaten und Doku

- [x] `CITATION.cff` anlegen.
  - Evidenz (2026-03-09): `CITATION.cff` liegt im Repo und wird im README als Zitierpfad referenziert.
- [x] `pyproject.toml` Authors/Maintainers explizit machen.
  - Evidenz (2026-03-09): `pyproject.toml` enthaelt explizite `authors`- und `maintainers`-Eintraege fuer Raphael Schlattmann.
- [x] `README.md` um Citation-, AND-, Scope- und Package-vs-Notebook-Abschnitte ergaenzen.
  - Evidenz (2026-03-09): `README.md` enthaelt Abschnitte zu `Audience and Scope`, `AND Integration Contract`, `Package vs Notebook Usage` und `How To Cite`.
- [x] Kein `.readthedocs.yaml` fuer `0.1.x`; `README.md` bleibt der zentrale Einstieg.
  - Evidenz (2026-03-09): `.readthedocs.yaml` ist nicht vorhanden; `README.md` ist der zentrale Doku-Einstiegspunkt im Repo.
- [ ] HF-API-Auth-/Env-Doku ergaenzen.
  - Status (2026-03-12): README/.env-Hinweise nennen bisher nur OpenRouter; fuer den hier verwendeten LiteLLM-HF-Pfad ist `HUGGINGFACE_API_KEY` der sicherste Env-Name, optional mit `HF_API_BASE`/`HUGGINGFACE_API_BASE` fuer abweichende Endpunkte.

## 7) Final Release Freeze

- [ ] Notebook-Outputs clearen und den Output-Cleanliness-Check explizit laufen lassen.
  - Status (2026-03-12): direkter Notebook-JSON-Check ergibt aktuell `11` Codezellen mit Outputs und `24` Codezellen mit `execution_count`; `git status --short` zeigt noch ein dirty `pipeline.ipynb`.
- [ ] `CHANGELOG.md` fuer den externen Release aktualisieren.
  - Status (2026-03-12): `CHANGELOG.md` steht weiter auf `Unreleased` und bildet die Aenderungen vom 2026-03-10/11 an Topic-Model-Backends, GGUF-Pooling und Notebook-/Runner-Refactors noch nicht ab.
- [ ] HF-Provider-Scope fuer `0.1.0` explizit entscheiden.
  - Status (2026-03-12): README bewirbt `huggingface_api`, aber die Tests decken aktuell nur Config-Freigabe plus einen Embedding-Unit-Test ab; es gibt weder ein Provider-Profil noch einen BERTopic-HF-API-Labeling-Test.
- [ ] Falls `huggingface_api` im Release-Scope bleibt: mindestens einen Smoke-Pfad plus einen manuellen/live Smoke nachziehen.
  - Status (2026-03-12): Die aktuellen Pipeline-Smokes sind offline/mocked und decken `openrouter`/`local`-Profile ab; lokale HF-Wege sind gut unit-getestet, aber nicht als echte Runtime-/Modell-Smokes abgesichert.
- [ ] Release-Metadaten finalisieren: Git-Tag, `python -m build`, `python -m twine check dist/*`, TestPyPI-Smoke.
  - Status (2026-03-12): `pyproject.toml` steht bereits auf `0.1.0`; `python -m build` und `python -m twine check dist/*` sind gruen; Git-Tag/Upload/TestPyPI-Smoke fehlen noch.
