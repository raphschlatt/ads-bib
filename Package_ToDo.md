# ADS Pipeline (`ads-bib`) - Package ToDo

Stand: 2026-02-20  
Ziel: Eine reale, projektspezifische Release-Checkliste fuer dieses Repo (nicht generisch).

## Ist-Stand (bereits gut)

- [x] `src`-Layout vorhanden (`src/ads_bib`).
- [x] Tests fuer zentrale Contracts vorhanden (u. a. Notebook- und Schema-Checks).
- [x] Daten-/Run-Skelett mit `.gitignore`-Regeln ist sauber angelegt (`data/`, `runs/`).

## 1) Release-Blocker (vor erstem offiziellen Release)

- [ ] **PEP-621 Metadaten in `pyproject.toml` vervollstaendigen**
  - Aktuell fehlen u. a. `readme`, `license`, `authors`, `classifiers`, `project.urls`.
  - Done when: `pyproject.toml` ist vollstaendig und konsistent gepflegt.

- [ ] **Lizenz festlegen und Datei hinzufuegen**
  - Aktuell keine `LICENSE`-Datei im Repo.
  - Done when: `LICENSE` vorhanden und `project.license` dazu passend gesetzt.

- [ ] **Root-`README.md` anlegen (PyPI/GitHub-tauglich)**
  - Aktuell gibt es keine README im Repo-Root.
  - Muss enthalten: Install, Quickstart, Extras, minimale API-Beispiele, externe API-Keys als Voraussetzungen.
  - Done when: README rendert sauber in `twine check`.

- [ ] **`CHANGELOG.md` einfuehren**
  - Done when: mindestens erster Eintrag fuer die naechste Release-Version vorhanden.

- [ ] **Abhaengigkeiten gegen echte Imports abgleichen**
  - Aktueller Befund:
  - `yaml` wird importiert (`src/ads_bib/run_manager.py`), `PyYAML` ist nicht deklariert.
  - `networkx` wird genutzt (`src/ads_bib/citations.py`), aber nicht deklariert.
  - `plotly` ist in `dependencies`, wird in `src/`/`tests/` derzeit nicht referenziert.
  - Done when:
  - direkte Imports sind in `dependencies` oder klar in Extras verankert.
  - optionale Teile haben entweder Lazy-Imports + klare Fehlermeldungen oder passende Extras.

- [ ] **CI Workflow mit Python-Matrix aufsetzen**
  - Aktuell keine Workflow-Datei unter `.github/workflows/`.
  - Minimum in CI:
  - `pytest -q`
  - `python -m build`
  - `twine check dist/*`
  - Done when: Matrix deckt mind. `min`/`max` Python aus `requires-python` ab (z. B. 3.10 + 3.12).

- [ ] **Wheel/sdist Installations-Smoketest ohne Repo-Pfad**
  - Warum: `tests/conftest.py` fuegt `src/` direkt zum `sys.path` hinzu; das ersetzt keinen echten Install-Test.
  - Done when: Build + frische venv + Import-Smoke laufen ohne `PYTHONPATH`-Tricks.

```bash
conda activate ADS_env
python -m build
python -m venv /tmp/ads_pkg_smoke
source /tmp/ads_pkg_smoke/bin/activate
WHEEL=$(ls dist/*.whl | head -n 1)
pip install "$WHEEL"
python -c "import ads_bib; print(ads_bib.__version__)"
python -c "import ads_bib.search, ads_bib.export, ads_bib.tokenize, ads_bib.translate, ads_bib.citations"
```

- [ ] **Extras-Smoketest fuer Topic/Visualisierung**
  - Done when: Topic-Stack installierbar und importierbar ueber Wheel-Extras.

```bash
source /tmp/ads_pkg_smoke/bin/activate
WHEEL=$(ls dist/*.whl | head -n 1)
pip install "$WHEEL[topic,umap,hdbscan]"
python -c "import ads_bib.topic_model, ads_bib.visualize"
```

## 2) Projekt-spezifische Release-Gates (aus AGENTS.md abgeleitet)

- [ ] **Notebook-Contract gruen halten**
  - `tests/test_pipeline_notebook_contract.py` muss gruen bleiben.
  - `pipeline.ipynb` ohne stale Outputs committen.

- [ ] **Schema-Kontrakte absichern**
  - `topic_id` statt `Cluster`.
  - `embedding_2d_x`/`embedding_2d_y` statt algorithmenspezifischer Namen.
  - Bei Schema-Aenderung: Tests fuer neue Spalten + alte Spalten-Nichtvorhandensein.

- [ ] **Toponymy/BERTopic Cost-Tracking-Namen stabil halten**
  - Erwartete Steps: `llm_labeling`, `llm_labeling_post_outliers`, `llm_labeling_toponymy`, `llm_labeling_toponymy_evoc`.
  - Compact Summary-Format in Kostenreports nicht aufbrechen.

- [ ] **Outlier-Refresh-Verhalten absichern**
  - Nach `reduce_outliers` muss `update_topics` weiterhin explizit passieren.

## 3) API und Kompatibilitaet (vor 1.0 festziehen)

- [ ] **Public API bewusst definieren**
  - Dokumentieren, welche Module/Funktionen stabil sind.
  - Optional: `__all__` und/oder API-Referenz entsprechend setzen.

- [ ] **Version als Single Source of Truth festlegen**
  - Aktuell doppelt gepflegt (`pyproject.toml` und `src/ads_bib/__init__.py`).
  - Done when: Drift ist ausgeschlossen (z. B. dynamisch aus Paketmetadaten oder klarer Release-Schritt).

## 4) Sicherheits- und Hygiene-Checks

- [ ] **Secrets-Check vor Tag/Release**
  - Keine Tokens/Keys im Repo, Notebook-Outputs oder Config-Snapshots.
  - `.env` bleibt ungetrackt.

- [ ] **Artefakt-Hygiene**
  - Keine grossen lokalen Outputs im Commit.
  - `data/` und `runs/` nur als Struktur, nicht mit Inhalten versionieren.

- [ ] **Dependency-Audit (pragmatisch)**
  - Mindestens einmal vor Release: `pip-audit` (oder CI-Scanner) laufen lassen und Ergebnis dokumentieren.

## 5) Finaler Release-Ablauf (fuer dieses Projekt)

- [ ] **Frischen Clone statt `git clean -xfd` verwenden**
  - Vermeidet versehentlichen Datenverlust bei lokalen Artefakten.

- [ ] **Release-Sequenz**
  - Version bump.
  - `python -m build`
  - Wheel/sdist Smoke + `pytest -q` + `twine check dist/*`
  - Git Tag `vX.Y.Z`
  - Release Notes aus `CHANGELOG.md`

- [ ] **Optional, aber stark empfohlen: TestPyPI**
  - Upload nach TestPyPI, dann Install + Smoke gegen TestPyPI-Paket.

## 6) Entscheidungen, die wir kurzfristig treffen muessen

- [ ] Lizenztyp (z. B. MIT/BSD-3-Clause/Apache-2.0).
- [ ] Offiziell unterstuetzte Python-Range (`>=3.10` beibehalten oder anpassen).
- [ ] Ob `networkx` Core-Dependency oder `citations`-Extra sein soll.
- [ ] Ob `plotly` entfernt wird oder in geplante Features klar eingeht.
