# ADS Pipeline (`ads-bib`) - Package ToDo

Stand: 2026-03-09
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

- [x] Mention-Input-Schema fuer den externen Runner festziehen: `mention_id`, `document_id`, `document_type`, `record_row`, `author_position`, `raw_mention`, optional `affiliation`, optional `year`.
  - Evidenz (2026-03-09): `src/ads_bib/author_disambiguation.py` baut genau dieses Mention-Schema; `tests/test_author_disambiguation.py` assertet die erwarteten Input-Spalten.
- [x] `apply_author_disambiguation()` als oeffentliche API einfuehren.
  - Evidenz (2026-03-09): Implementiert in `src/ads_bib/author_disambiguation.py`, re-exportiert via `src/ads_bib/__init__.py`; `tests/test_package_exports.py` deckt den Public Export ab.
- [x] Rueckgabevertrag des externen Runners validieren: `mention_assignments` plus `authors`.
  - Evidenz (2026-03-09): `src/ads_bib/author_disambiguation.py` validiert Pflichtspalten, Nullwerte, Duplicate-IDs und Mention-Abdeckung fuer `mention_assignments` sowie Pflichtspalten/IDs fuer `authors`.
- [x] Rueck-Mapping nach `publications` und `references` implementieren.
  - Zielspalten: `author_uids` und `author_display_names`, positions-aligned zu `Author`.
  - Evidenz (2026-03-09): `_apply_assignments_to_frame()` schreibt positions-ausgerichtete `author_uids`/`author_display_names`; `tests/test_author_disambiguation.py` deckt Rueck-Mapping fuer Publications und References ab.
- [x] Phase-4-Checkpoints in Parquet einfuehren.
  - Dateien: `publications_disambiguated.parquet`, `references_disambiguated.parquet`, `authors.parquet`.
  - Evidenz (2026-03-09): `src/ads_bib/_utils/checkpoints.py` implementiert `save_phase4_checkpoint()`/`load_phase4_checkpoint()`; `tests/test_checkpoints.py` deckt Schreiben, Laden und Run-Snapshot ab.
- [ ] Tests mit gemocktem externen Runner einfuehren.
  - Status (2026-03-09): Mapping, Cache-Reload und Run-Snapshot sind per Mock abgesichert; explizite Negativtests fuer fehlerhafte `mention_assignments`/`authors` fehlen noch.
- [ ] Referenzielle Konsistenz zwischen `mention_assignments.author_uid` und `authors.author_uid` explizit validieren.
  - Status (2026-03-09): Der Vertrag setzt aktuell voraus, dass alle referenzierten `author_uid` auch in `authors` vorkommen; diese Cross-Frame-Validierung ist noch nicht implementiert.

## 3) Anforderungen an das externe AND-Package

- [ ] Generische Mention-basierte API dokumentieren; keine ADS-DataFrames als externe Package-API.
  - Status (2026-03-09): Erwartung ist auf `ads-bib`-Seite in `README.md` dokumentiert; die verbindliche Festschreibung im externen AND-Package bleibt offen.
- [ ] Rueckgabe von `mention_assignments` plus `authors` als verbindlichen Vertrag beschreiben.
  - Status (2026-03-09): Rueckgabeformat ist in `README.md` und `src/ads_bib/author_disambiguation.py` beschrieben bzw. validiert; der formale Vertrag des externen Packages ist noch nicht festgezogen.
- [ ] `author_display_name` als generische Entity-Metadatenlogik im externen Package festhalten.
  - Heuristik gehoert dort hin, nicht in `ads-bib`.
  - Status (2026-03-09): `README.md` verortet die menschenlesbare Repraesentationslogik bewusst im externen AND-Package; die tatsaechliche Umsetzung dort bleibt offen.

## 4) Citation-Aufraeumen

- [x] `author_co_citation` auf eigene Author-Nodes umstellen.
  - Evidenz (2026-03-09): `src/ads_bib/citations.py` baut fuer `author_co_citation` dedizierte Author-Nodes via `_build_author_nodes()`; `tests/test_citations_additional.py` assertet Author-Node-Exporte.
- [x] `author_uids` bevorzugen, Fallback auf bestehende First-Author-Name-Logik behalten.
  - Evidenz (2026-03-09): `_resolve_first_author_id()` priorisiert `author_uids` und faellt sonst auf `_first_author_lastname()` zurueck; `tests/test_citations_author_list.py` deckt UID-Priorisierung ab.
- [x] Labels aus `author_entities` ziehen, wenn vorhanden.
  - Evidenz (2026-03-09): `create_author_co_citations()` und `_build_author_nodes()` ziehen Labels aus `author_entities`; `tests/test_citations_author_list.py` deckt `author_display_name`-Labels ab.
- [x] Keine gemischten Bibcode-/Author-Knoten mehr exportieren.
  - Evidenz (2026-03-09): `process_all_citations(..., metric=\"author_co_citation\")` exportiert nur Author-Nodes fuer diese Metrik; `tests/test_citations_additional.py` assertet ausschliesslich Author-IDs in `nodes.id`.

## 5) Release-Hygiene

- [x] `.claude/settings.local.json` aus dem Repo entfernen und `.claude/` ignorieren.
  - Evidenz (2026-03-09): `.gitignore` enthaelt `.claude/`; `git ls-files .claude/settings.local.json .claude` liefert keine Treffer.
- [x] `archive/`, `AGENTS.md`, `CLAUDE.md` und `Package_ToDo.md` aus SDist/Wheel ausschliessen.
  - Evidenz (2026-03-09): `pyproject.toml` schliesst diese Pfade im SDist aus; Artefaktpruefung nach `conda run -n ADS_env python -m build` ergibt im SDist keine Treffer und im Wheel `NO_MATCHES`.
- [ ] `archive/` aus dem Default-Branch vor externem Release neu entscheiden.
  - Status (2026-03-09): Artefakt-Ausschluss ist umgesetzt; der Verbleib von `archive/` im Repo ist noch eine gesonderte Release-Entscheidung.
- [x] `.gitignore` explizit um Coverage-/HTML-/Ruff-Artefakte ergaenzen.
  - Evidenz (2026-03-09): `.gitignore` enthaelt `.ruff_cache/`, `.coverage`, `.coverage.*` und `htmlcov/`.
- [x] SDist/Wheel-Inhalte nach jedem Packaging-Change pruefen.
  - Erfolgskriterium: keine lokalen Tooling-Dateien und keine internen Backlogs im Artefakt.
  - Evidenz (2026-03-09): `conda run -n ADS_env python -m build` -> sdist und wheel erfolgreich gebaut; `conda run -n ADS_env python -m twine check dist/*` -> beide Artefakte `PASSED`; Inhaltspruefungen fuer SDist/Wheel sind sauber.

## 6) Metadaten und Doku

- [x] `CITATION.cff` anlegen.
  - Evidenz (2026-03-09): `CITATION.cff` liegt im Repo und wird im README als Zitierpfad referenziert.
- [x] `pyproject.toml` Authors/Maintainers explizit machen.
  - Evidenz (2026-03-09): `pyproject.toml` enthaelt explizite `authors`- und `maintainers`-Eintraege fuer Raphael Schlattmann.
- [x] `README.md` um Citation-, AND-, Scope- und Package-vs-Notebook-Abschnitte ergaenzen.
  - Evidenz (2026-03-09): `README.md` enthaelt Abschnitte zu `Audience and Scope`, `AND Integration Contract`, `Package vs Notebook Usage` und `How To Cite`.
- [x] Kein `.readthedocs.yaml` fuer `0.1.x`; `README.md` bleibt der zentrale Einstieg.
  - Evidenz (2026-03-09): `.readthedocs.yaml` ist nicht vorhanden; `README.md` ist der zentrale Doku-Einstiegspunkt im Repo.

## 7) Final Release Freeze

- [ ] Notebook-Outputs clearen und den Output-Cleanliness-Check explizit laufen lassen.
  - Status (2026-03-09): direkter Notebook-JSON-Check ergibt aktuell `not_clean`; der explizite Release-Freeze-Schritt bleibt offen.
- [ ] `CHANGELOG.md` fuer den externen Release aktualisieren.
  - Status (2026-03-09): `CHANGELOG.md` steht auf `Unreleased` und bildet den externen Release-Stand fuer AND/Citation/Packaging noch nicht final ab.
- [ ] Version bump, `python -m build`, `python -m twine check dist/*`, TestPyPI-Smoke.
  - Status (2026-03-09): `python -m build` und `python -m twine check dist/*` sind gruen; Version bump und TestPyPI-Smoke fehlen noch.
