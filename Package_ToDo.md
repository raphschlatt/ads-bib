# ADS Pipeline (`ads-bib`) - Package ToDo

Stand: 2026-03-06
Ziel: erster externer Release eines notebook-first Research-Packages mit externem AND-Adapter.

## Festgelegte Entscheidungen

- Kein `uv.lock` in dieser Release-Welle; `conda activate ADS_env` bleibt der kanonische lokale Pfad.
- AI-Dateien duerfen im Repo bleiben, aber nicht in SDist/Wheel landen; lokale `.claude/`-Dateien gehoeren nicht ins Repo.
- AND bleibt ein externes Package; `ads-bib` baut nur den ADS-spezifischen Adapter, die Validierung, das Rueck-Mapping und die Downstream-Nutzung.
- Notebook output-clean ist ein Release-Freeze-Thema und kein taeglicher Development-Gate.
- `CITATION.cff` kommt rein; ein separates `AUTHORS.md` ist fuer `0.1.x` nicht erforderlich.

## 1) Sofortige Korrekturen

- [ ] `filter_by_field()` auf exaktes, case-insensitive Matching fuer Stringspalten ziehen.
  - Erfolgskriterium: `tests/test_curate.py` ist gruen und kein Regex-/Substring-Verhalten bleibt aktiv.
- [ ] `ads-bib check` auf interpreter-lokale Modulaufrufe (`python -m ...`) umstellen.
  - Erfolgskriterium: der Windows-Wrapper-Fehler tritt nicht mehr auf.
- [ ] Notebook-Cleanliness aus den Daily-Gates loesen, aber den Release-Check erhalten.
  - Erfolgskriterium: `python -m pytest -q` ist gruen, ohne dass `pipeline.ipynb` gecleart sein muss.
- [ ] Pandas-Warnung in `export.py` fuer `select_dtypes(include="object")` entfernen.
  - Erfolgskriterium: keine entsprechende Warnung mehr in der Testsuite.

## 2) AND-Integration in diesem Repo

- [ ] Mention-Input-Schema fuer den externen Runner festziehen: `mention_id`, `document_id`, `document_type`, `record_row`, `author_position`, `raw_mention`, optional `affiliation`, optional `year`.
- [ ] `apply_author_disambiguation()` als oeffentliche API einfuehren.
- [ ] Rueckgabevertrag des externen Runners validieren: `mention_assignments` plus `authors`.
- [ ] Rueck-Mapping nach `publications` und `references` implementieren.
  - Zielspalten: `author_uids` und `author_display_names`, positions-aligned zu `Author`.
- [ ] Phase-4-Checkpoints in Parquet einfuehren.
  - Dateien: `publications_disambiguated.parquet`, `references_disambiguated.parquet`, `authors.parquet`.
- [ ] Tests mit gemocktem externen Runner einfuehren.
  - Erfolgskriterium: Mapping, Validierung, Cache-Reload und Run-Snapshot sind abgesichert.

## 3) Anforderungen an das externe AND-Package

- [ ] Generische Mention-basierte API dokumentieren; keine ADS-DataFrames als externe Package-API.
- [ ] Rueckgabe von `mention_assignments` plus `authors` als verbindlichen Vertrag beschreiben.
- [ ] `author_display_name` als generische Entity-Metadatenlogik im externen Package festhalten.
  - Heuristik gehoert dort hin, nicht in `ads-bib`.

## 4) Citation-Aufraeumen

- [ ] `author_co_citation` auf eigene Author-Nodes umstellen.
- [ ] `author_uids` bevorzugen, Fallback auf bestehende First-Author-Name-Logik behalten.
- [ ] Labels aus `author_entities` ziehen, wenn vorhanden.
- [ ] Keine gemischten Bibcode-/Author-Knoten mehr exportieren.
  - Erfolgskriterium: Author-Co-Citation-Exporte enthalten nur Author-Nodes.

## 5) Release-Hygiene

- [ ] `.claude/settings.local.json` aus dem Repo entfernen und `.claude/` ignorieren.
- [ ] `archive/`, `AGENTS.md`, `CLAUDE.md` und `Package_ToDo.md` aus SDist/Wheel ausschliessen.
- [ ] `archive/` aus dem Default-Branch vor externem Release neu entscheiden.
  - Minimum fuer diese Welle: nicht mehr in SDist/Wheel.
- [ ] `.gitignore` explizit um Coverage-/HTML-/Ruff-Artefakte ergaenzen.
- [ ] SDist/Wheel-Inhalte nach jedem Packaging-Change pruefen.
  - Erfolgskriterium: keine lokalen Tooling-Dateien und keine internen Backlogs im Artefakt.

## 6) Metadaten und Doku

- [ ] `CITATION.cff` anlegen.
- [ ] `pyproject.toml` Authors/Maintainers explizit machen.
- [ ] `README.md` um Citation-, AND-, Scope- und Package-vs-Notebook-Abschnitte ergaenzen.
- [ ] Kein `.readthedocs.yaml` fuer `0.1.x`; `README.md` bleibt der zentrale Einstieg.

## 7) Final Release Freeze

- [ ] Notebook-Outputs clearen und den Output-Cleanliness-Check explizit laufen lassen.
- [ ] `CHANGELOG.md` fuer den externen Release aktualisieren.
- [ ] Version bump, `python -m build`, `python -m twine check dist/*`, TestPyPI-Smoke.
