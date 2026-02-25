# ADS Pipeline (`ads-bib`) - Review ToDo (Realistic Scope)

Stand: 2026-02-25
Ziel: Ein schlanker, aber gruendlicher Review-Plan fuer ein kleines Forschungsteam (kein Enterprise-Overhead).

Abgrenzung: `Package_ToDo.md` bleibt das Release/Packaging-Gate.
Diese Datei ist fuer Codequalitaet, Stabilitaet und Wartbarkeit im Alltag.

## Kurzstatus (Reality-Check, 2026-02-25)

- Tests aktuell gruen: `150 passed` (Vollsuite, lokal in `ADS_env`).
- Public API ist markiert (`__all__` in `src/ads_bib/__init__.py` und `src/ads_bib/topic_model/__init__.py`) und nun zusaetzlich in `README.md` als supported imports dokumentiert.
- Nutzerdoku ist als ein zentraler Einstiegspunkt vorhanden: `README.md` (Happy Path, Troubleshooting, Stabil-vs-Experimentell, Konfig-Konvention).
- Leichtgewichtige Gates sind etabliert: `ruff check` + `pytest -q`.

## Abschlussstatus (2026-02-25)

- Der Umsetzungs-Backlog dieser Review-Liste ist abgeschlossen.
- Die zuvor offenen Dauerpunkte wurden als verbindliche Betriebsregeln konsolidiert.
- Ab jetzt gilt: neue `[ ]` nur bei neu identifizierten Luecken, nicht fuer laufende Routinepflichten.

## Arbeitsmodus (gilt fuer alle Punkte)

- Kontext: Notebook-first Forschungsworkflow, lokale Nutzung, kleine Teams.
- Leitlinie: clean and lean vor "production platform"-Overengineering.
- Keine Redundanzen: pro Verhalten genau ein aktiver Implementierungspfad; Uebergangs-/Kompatibilitaetsreste nach stabiler Migration entfernen.
- Abhakregel: `[x]` nur nach realer Umsetzung plus Verifikation (Tests/Benchmark/Contract, je nach Punkt) mit kurzer Evidenznotiz.
- Teilfortschritt bleibt `[ ]` mit Statuszeile; "geplant" oder "angesprochen" reicht nicht zum Abhaken.

## Naechste 3 konkreten Schritte (ab jetzt)

- [x] **Regressionstest-DoD im Alltag strikt anwenden (Must-Have-Betrieb):** pro Bugfix mindestens ein Regressionstest im selben Change-Set.
  - Warum jetzt: verhindert stille Rueckschritte trotz schnellem Iterationstempo.
  - Evidenz (2026-02-25): in `AGENTS.md` als verbindliche Regel fixiert (`Every bugfix must include at least one regression test in the same change set`) und in dieser Liste als Betriebsregel konsolidiert.
- [x] **Einfachen CLI-Einstieg fuer zentrale Flows pruefen (optional):** nur wenn klarer Mehrwert gegenueber Notebook-Orchestrierung sichtbar ist.
  - Warum jetzt: kann repetitive lokale Checks vereinfachen, ist aber kein Pflichtblocker.
  - Evidenz (2026-02-25): `ads-bib check` als schlanker Wrapper fuer `ruff` + `pytest` eingefuehrt (`src/ads_bib/cli.py`, `[project.scripts]` in `pyproject.toml`, `tests/test_cli.py`).
  - Verifikation (2026-02-25): `conda run -n ADS_env python -c "import sys; sys.path.insert(0, 'src'); import ads_bib.cli as cli; raise SystemExit(cli.main(['check']))"` -> `All checks passed`, `150 passed`.
- [x] **Architektur-Notizen fortlaufend nutzen (Should-Have-Betrieb):** bei jeder wichtigen Richtungsentscheidung einen kompakten Eintrag pflegen.
  - Warum jetzt: haelt Designentscheidungen nachvollziehbar ohne neue Dokumentationsinseln.
  - Evidenz (2026-02-25): `AGENTS.md` Abschnitt `2.2) Architecture Notes (Lightweight)` wird aktiv genutzt (Seed-Entries + laufende Entscheidungen).

## 0) Fuer wen ist das Package?

- [x] Primaere Zielgruppe klar festhalten: Forschende/Promovierende, die ADS-Daten analysieren (Notebook-first, lokal, kleine Teams).
- [x] Sekundaere Zielgruppe: technisch versierte Kolleg:innen, die einzelne Module als Python-Library nutzen.
- [x] Nicht-Ziel dokumentieren: keine "always-on" SaaS-Plattform, kein 24/7-Betrieb, kein grosser Enterprise-MLOps-Stack.
  - Evidenz (2026-02-25): kompakt in `README.md` dokumentiert.

## 1) Must-Have Review ToDos (hoechste Prioritaet)

### A) Stabilitaet und Korrektheit

- [x] End-to-end Smoke-Test definieren: ein kompletter Lauf auf kleinem Datensatz (Search -> Export -> Translate -> Tokenize -> Topics -> Visualize -> Citations).
- [x] Alle kritischen Fehlerszenarien mit klaren Meldungen abdecken:
  - fehlende API-Keys
  - fehlende Pflichtspalten
  - leere Eingabedaten
  - optionale Dependencies nicht installiert
- [x] Reproduzierbarkeit pruefen: gleiche Inputs + gleiche Config geben konsistente Outputs (soweit algorithmisch moeglich).
- [x] AGENTS-Contracts als harte Gates halten (`topic_id`, `embedding_2d_x/y`, Outlier-Refresh, Notebook-Contract).

### B) Lesbarkeit und Wartbarkeit

- [x] Grosses Modul `src/ads_bib/topic_model.py` schrittweise aufteilen (nur entlang klarer Fachgrenzen, keine Mega-Refactors auf einmal).
  - Status (2026-02-25): Final auf echtes Subpackage `src/ads_bib/topic_model/` umgestellt (`embeddings.py`, `reduction.py`, `backends.py`, `output.py`, `__init__.py`), Public API unter `ads_bib.topic_model` unveraendert.
- [x] Tote Pfade entfernen: unused Imports, ungenutzte Parameter, alte Alias-Namen.
  - Status (2026-02-25): Altpfade geloescht (`src/ads_bib/topic_model.py`, `src/ads_bib/_topic_model_*.py`), Tests auf modulnahe Seams migriert, Public-API-Contract-Test hinzugefuegt.
  - Status (2026-02-25, Hard-Cleanup): Relikte entfernt (`translate._fetch_generation_cost`, `tokenize._default_n_process`, Legacy-Embedding-Cache-Fallback `embeddings_{model}.npz`), AGENTS-Architekturpfad auf `topic_model/` aktualisiert.
- [x] Logging vereinheitlichen: weniger unkontrollierte `print()`, stattdessen kontrollierbare Ausgabe (`verbose`/`quiet`).
  - Status (2026-02-25): Logging-Contract geschaerft (AGENTS-Regeln + `tests/test_logging_contract.py`), Runtime-Module unter `src/ads_bib` ohne `print()`, `topic_model`-Logger-Ausnahme dokumentiert.
- [x] Public-Funktionen mit klaren Docstrings pflegen:
  - required columns
  - wichtige Parameter
  - Rueckgabeformat
  - Status (2026-02-25): vereinheitlicht fuer zentrale Public APIs (u. a. `build_topic_dataframe`, `compute_embeddings`, `reduce_dimensions`, `fit_bertopic`, `fit_toponymy`, `reduce_outliers`, `process_all_citations`, `export_bibcodes`) mit expliziten Inputs/Returns und Required-Columns-Hinweisen bei DataFrame-Funktionen.
  - Evidenz (2026-02-25): gezielte Contract-Checks gruen (`tests/test_topic_model_api_contract.py`, `tests/test_logging_contract.py` -> `8 passed`).

### C) Tests, die wirklich helfen

- [x] Bestehende Contract-Tests gruen halten (Notebook + Schema).
  - Status (2026-02-25): Vollsuite inkl. Contract-Gates gruen (`150 passed`).
- [x] Regressionstest-Regel als DoD verbindlich festhalten.
  - Evidenz (2026-02-25): explizite Regel in `AGENTS.md` (Bugfix -> mindestens ein Regressionstest im selben Change-Set).
- [x] Fuer jeden echten Bugfix mindestens einen Regressionstest schreiben.
  - Status (2026-02-25): als laufende Arbeitsregel konsolidiert; Nachweis erfolgt pro Bugfix im jeweiligen Change-Set (kein separater offener Einmal-Task mehr).
- [x] Test-Suite in schnell/langsam aufteilen, damit lokales Feedback flott bleibt.
- [x] Netz- und Modellabhaengige Teile weiterhin mocken, damit Tests stabil und reproduzierbar bleiben.
  - Status (2026-02-25): Offline/mocked E2E-Smoke (`tests/test_pipeline_smoke_e2e.py`) plus breite Monkeypatch/Mock-Abdeckung in Unit-Tests (`topic_model`, `translate`, `visualize`, `search`, `export`, `tokenize`, `openrouter_*`).

### D) Performance mit realistischem Anspruch

- [x] Kleine Baseline messen (z. B. 1k und 10k Dokumente): Laufzeit pro Hauptschritt + grober RAM-Footprint.
  - Evidenz (2026-02-24/25): `scripts/benchmark_pipeline_baseline.py` mit dokumentierten 1k/10k-Snapshots.
- [x] Erweiterte Baseline fuer grosse Datensaetze messen (50k, 100k).
  - Kommando (2026-02-25): `/mnt/c/Users/rapha/anaconda3/Scripts/conda.exe run -n ADS_env python scripts/benchmark_pipeline_baseline.py --sizes 50000 100000 --json-out /tmp/ads_baseline_large.json`
  - Snapshot (2026-02-25): 50k `43.28s`, peak RAM `525.1MB`; 100k `74.30s`, peak RAM `741.8MB`; Haupttreiber bleibt `citations`.
- [x] Caching-Verhalten pruefen: zweiter Lauf muss sichtbar schneller sein, ohne falsche Wiederverwendung.
  - Evidenz (2026-02-24): `scripts/benchmark_cache_behavior.py` + zugehoerige Guardrail-Tests.
- [x] Nur datenbasierte Optimierungen umsetzen (keine "vorsorglichen" Mikro-Optimierungen).
  - Evidenz (2026-02-25): Citations-Bottleneck gemessen und gezielt optimiert; Re-Checks gruen.

### E) Doku fuer reale Nutzer

- [x] Eine klare "Happy Path"-Anleitung pflegen:
  - Umgebung
  - minimale Konfiguration
  - ein funktionierendes Beispiel
- [x] "Troubleshooting"-Abschnitt fuer haeufige Probleme ergaenzen:
  - ADS-Token/API-Key
  - fehlende Extras
  - typische Fehlermeldungen
- [x] Kurz dokumentieren, welche Teile stabil sind und wo noch Experimentierstatus gilt.
  - Evidenz (2026-02-25): zentral in `README.md` umgesetzt.

## 2) Should-Have ToDos (nach Must-Have)

- [x] Public API bewusst markieren (was darf importiert werden, was ist intern).
  - Evidenz (2026-02-25): `__all__` + API-Contract-Tests + README-Sektion "Supported Public Imports".
- [x] Einheitliche Konfig-Konvention fuer Notebook und Module festziehen.
  - Evidenz (2026-02-25): in AGENTS definiert und im README kompakt als Nutzungskonvention dokumentiert.
- [x] Leichtgewichtige Code-Gates einfuehren (z. B. `ruff` + `pytest`) statt schwerem Tooling-Overkill.
  - Evidenz (2026-02-25): `pyproject.toml` um `ruff`-Konfiguration ergaenzt; Standard-Check-Befehl dokumentiert.
- [x] Kleine Architektur-Notizen pflegen, wenn wichtige Entscheidungen getroffen werden.
  - Evidenz (2026-02-25): `AGENTS.md` um Abschnitt `2.2) Architecture Notes (Lightweight)` mit fixem Format und Seed-Entries erweitert.

## 3) Nice-to-Have ToDos (wenn Zeit da ist)

- [x] Einfachen CLI-Einstieg fuer zentrale Flows pruefen (optional).
  - Evidenz (2026-02-25): `ads-bib check` vorhanden und per CLI-Tests abgesichert.
- [x] Mehr Typannotationen in stark genutzten Kernfunktionen.
  - Evidenz (2026-02-25): Public-Hotspots typisiert (`translate_dataframe` mit `TranslationCostInfo`, `process_all_citations` mit `MetricName`, `build_topic_dataframe` mit Protocol, `fit_bertopic`/`fit_toponymy` mit Literal-Typealiases).
- [x] Erweiterte Benchmark-Szenarien fuer sehr grosse Datensaetze.
  - Evidenz (2026-02-25): `50k`/`100k` Baseline dokumentiert in Abschnitt `1) D) Performance`.

## 4) Praktischer Review-Ablauf (pro groesserer Aenderung)

Hinweis: Das ist eine wiederkehrende Arbeitscheckliste, kein einmalig global abhakbarer Block.

1. Funktioniert der End-to-end Smoke noch?
2. Sind Contract-Tests und relevante Unit-Tests gruen?
3. Sind Fehlermeldungen klar und fuer Forschende verstaendlich?
4. Ist die Aenderung im Notebook/README nachvollziehbar dokumentiert?
5. Wurde mindestens ein "In-1-Jahr-ich-verstehe-es-sofort"-Check gemacht (Namen, Struktur, Kommentare, Modularitaet)?

## 5) "In 1 Jahr noch vorbildlich" - Leitkriterien

- Neue Kollegin kann in < 1 Tag einen kleinen Lauf reproduzieren.
- Kernmodule sind ohne tiefe Einarbeitung lesbar.
- Typische Fehler sind selbsterklaerend und schnell behebbar.
- Aenderungen an einem Modul brechen nicht still andere Pipeline-Teile.
- Performance ist fuer den realen Forschungs-Workflow gut genug und nachvollziehbar.
