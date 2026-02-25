# ADS Pipeline (`ads-bib`) - Review ToDo (Realistic Scope)

Stand: 2026-02-25  
Ziel: Ein schlanker, aber gruendlicher Review-Plan fuer ein kleines Forschungsteam (kein Enterprise-Overhead).

Abgrenzung: `Package_ToDo.md` bleibt das Release/Packaging-Gate.  
Diese Datei ist fuer Codequalitaet, Stabilitaet und Wartbarkeit im Alltag.

## Kurzstatus (Reality-Check, 2026-02-25)

- Tests aktuell gruen: `147 passed` (Vollsuite, lokal in `ADS_env`).
- Public API ist teilweise bereits markiert (`__all__` in `src/ads_bib/__init__.py` und `src/ads_bib/topic_model/__init__.py`), aber nicht als kurze Nutzer-Policy dokumentiert.
- Es gibt aktuell kein `README.md`/`docs/` im Repo-Root; die offenen Nutzer-Doku-Punkte bleiben daher hoch relevant.
- Leichtgewichtige Gates sind teilweise da (`pytest`-Config in `pyproject.toml`), `ruff`/Lint-Gate fehlt noch.

## Arbeitsmodus (gilt fuer alle Punkte)

- Kontext: Notebook-first Forschungsworkflow, lokale Nutzung, kleine Teams.
- Leitlinie: clean and lean vor "production platform"-Overengineering.
- Keine Redundanzen: pro Verhalten genau ein aktiver Implementierungspfad; Uebergangs-/Kompatibilitaetsreste nach stabiler Migration entfernen.
- Abhakregel: `[x]` nur nach realer Umsetzung plus Verifikation (Tests/Benchmark/Contract, je nach Punkt) mit kurzer Evidenznotiz.
- Teilfortschritt bleibt `[ ]` mit Statuszeile; "geplant" oder "angesprochen" reicht nicht zum Abhaken.

## Naechste 3 konkreten Schritte (im Projektkontext priorisiert)

- [ ] **Doku-Basis fuer reale Nutzung erstellen (Must-Have):** kurzer Happy Path + Troubleshooting + Stabil-vs-Experimentell.
  - Warum jetzt: Notebook-first Forschungsteam braucht reproduzierbaren Einstieg ohne tiefe Code-Lektuere.
- [ ] **Public-Docstrings finalisieren (Must-Have):** bei allen Public-Funktionen required columns, Schluesselparameter und Rueckgabeformat explizit halten.
  - Warum jetzt: reduziert Rueckfragen in der Notebook-Orchestrierung und stabilisiert Modulgrenzen.
- [ ] **Review-Gates formalisieren (Should-Have):** `pytest` ist etabliert, jetzt schlankes Lint-Gate (`ruff`) nachziehen.
  - Warum jetzt: geringe Zusatzlast, aber klare Qualitaetsgrenze fuer weitere Refactors.

## 0) Fuer wen ist das Package?

- [ ] Primaere Zielgruppe klar festhalten: Forschende/Promovierende, die ADS-Daten analysieren (Notebook-first, lokal, kleine Teams).
- [ ] Sekundaere Zielgruppe: technisch versierte Kolleg:innen, die einzelne Module als Python-Library nutzen.
- [ ] Nicht-Ziel dokumentieren: keine "always-on" SaaS-Plattform, kein 24/7-Betrieb, kein grosser Enterprise-MLOps-Stack.
  - Status (2026-02-25): Kontext ist in `AGENTS.md` bereits klar, aber noch nicht in einer kurzen nutzerorientierten Einstiegsdoku festgehalten.

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
  - Status (2026-02-25): Logging-Contract geschärft (AGENTS-Regeln + `tests/test_logging_contract.py`), Runtime-Module unter `src/ads_bib` ohne `print()`, `topic_model`-Logger-Ausnahme dokumentiert.
- [x] Public-Funktionen mit klaren Docstrings pflegen:
  - required columns
  - wichtige Parameter
  - Rueckgabeformat
  - Status (2026-02-25): vereinheitlicht fuer zentrale Public APIs (u. a. `build_topic_dataframe`, `compute_embeddings`, `reduce_dimensions`, `fit_bertopic`, `fit_toponymy`, `reduce_outliers`, `process_all_citations`, `export_bibcodes`) mit expliziten Inputs/Returns und Required-Columns-Hinweisen bei DataFrame-Funktionen.
  - Evidenz (2026-02-25): gezielte Contract-Checks gruen (`tests/test_topic_model_api_contract.py`, `tests/test_logging_contract.py` -> `8 passed`).

### C) Tests, die wirklich helfen

- [x] Bestehende Contract-Tests gruen halten (Notebook + Schema).
  - Status (2026-02-25): Vollsuite inkl. Contract-Gates gruen (`147 passed`).
- [ ] Fuer jeden echten Bugfix mindestens einen Regressionstest schreiben.
  - Status (2026-02-25): bleibt als laufende Arbeitsregel offen; wird pro Bugfix im jeweiligen Change-Set evidenziert.
- [x] Test-Suite in schnell/langsam aufteilen, damit lokales Feedback flott bleibt.
- [x] Netz- und Modellabhaengige Teile weiterhin mocken, damit Tests stabil und reproduzierbar bleiben.
  - Status (2026-02-25): Offline/mocked E2E-Smoke (`tests/test_pipeline_smoke_e2e.py`) plus breite Monkeypatch/Mock-Abdeckung in Unit-Tests (`topic_model`, `translate`, `visualize`, `search`, `export`, `tokenize`, `openrouter_*`).

### D) Performance mit realistischem Anspruch

- [x] Kleine Baseline messen (z. B. 1k und 10k Dokumente): Laufzeit pro Hauptschritt + grober RAM-Footprint. Ist es RAM sparsam auch für große Datasets?
  - Benchmark-Runner: `scripts/benchmark_pipeline_baseline.py`
  - Kommando: `PYTHONPATH=src /mnt/c/Users/rapha/anaconda3/envs/ADS_env/python.exe scripts/benchmark_pipeline_baseline.py --sizes 1000 10000`
  - Snapshot (2026-02-24, offline/mocked full pipeline inkl. citations metrics=`direct,co_citation,bibliographic_coupling,author_co_citation`):
    - 1k docs: total `3.28s`, peak RAM `303.4 MB`
      - search `0.01s`, export `0.09s`, translate `0.08s`, tokenize `0.01s`, topics `0.01s`, visualize `1.36s`, citations `1.72s`
    - 10k docs: total `10.22s`, peak RAM `347.9 MB`
      - search `0.03s`, export `0.88s`, translate `0.91s`, tokenize `0.20s`, topics `0.04s`, visualize `1.53s`, citations `6.62s`
  - Einordnung: fuer diese Baseline wirkt der Speicherbedarf fuer 10k weiterhin moderat (Peak +44.5 MB ggü. 1k); Hauptkostentreiber ist `citations`.
  - Snapshot (2026-02-25, nach Topic-Model-Reset):
    - 1k docs: total `2.73s`, peak RAM `302.2 MB`
    - 10k docs: total `9.11s`, peak RAM `349.9 MB`
  - Vergleich zur Pre-Reset-Baseline (2026-02-25, `2.59s`/`8.17s`): leichte Mehrzeit (`+0.14s` bei 1k, `+0.94s` bei 10k), RAM praktisch unveraendert; keine grobe Regression.
- [x] Caching-Verhalten pruefen: zweiter Lauf muss sichtbar schneller sein, ohne falsche Wiederverwendung.
  - Benchmark-Runner: `scripts/benchmark_cache_behavior.py`
  - Kommando: `PYTHONPATH=src /mnt/c/Users/rapha/anaconda3/envs/ADS_env/python.exe scripts/benchmark_cache_behavior.py --docs 10000 --delay 0.15`
  - Snapshot (2026-02-24):
    - Embeddings cache: cold `0.188s` -> warm `0.008s` (`22.46x` schneller)
    - Reduction cache: cold `0.352s` -> warm `0.006s` (`56.75x` schneller)
    - Invalidation geprueft:
      - Input-Aenderung bei Embeddings triggert Recompute (`backend_calls=2`, kein falscher Cache-Hit)
      - Parameter-Aenderung bei Reduction triggert 5D-Recompute, 2D bleibt korrekt im Cache (`backend_fit_calls=3`)
  - Automated Guardrails:
    - `tests/test_topic_model.py::test_compute_embeddings_uses_cache_on_second_call`
    - `tests/test_topic_model.py::test_reduce_dimensions_uses_cache_then_recomputes_on_param_change`
- [x] Nur datenbasierte Optimierungen umsetzen (keine "vorsorglichen" Mikro-Optimierungen).
  - Status (2026-02-25): Citations wurde als Bottleneck erst gemessen und dann gezielt optimiert (sparse Pfade). Re-Checks: Vollsuite `138 passed`; Baseline 10k weiterhin mit `citations` als Haupttreiber (~`8.9-9.1s`) und ohne unkontrollierte Seiteneffekte.
  - Status-Update (2026-02-25): aktueller Re-Check steht bei `147 passed`; Performance-Einordnung unveraendert.

### E) Doku fuer reale Nutzer

- [ ] Eine klare "Happy Path"-Anleitung pflegen:
  - Umgebung
  - minimale Konfiguration
  - ein funktionierendes Beispiel
  - Status (2026-02-25): noch offen; es gibt derzeit kein zentrales `README.md`/`docs/` als Einstieg.
- [ ] "Troubleshooting"-Abschnitt fuer haeufige Probleme ergaenzen:
  - ADS-Token/API-Key
  - fehlende Extras
  - typische Fehlermeldungen
  - Status (2026-02-25): noch offen; Inhalte sind teils implizit in Code/Tests, aber nicht als Nutzerhilfe gebuendelt.
- [ ] Kurz dokumentieren, welche Teile stabil sind und wo noch Experimentierstatus gilt.
  - Status (2026-02-25): noch offen; technische Stabilitaet ist gut, aber Stabilitaetsgrad je Modul noch nicht explizit publiziert.

## 2) Should-Have ToDos (nach Must-Have)

- [ ] Public API bewusst markieren (was darf importiert werden, was ist intern).
  - Status (2026-02-25): teilweise erledigt (`__all__` + API-Contract-Tests), aber klare "supported imports"-Notiz fuer Nutzer fehlt.
- [ ] Einheitliche Konfig-Konvention fuer Notebook und Module festziehen.
  - Status (2026-02-25): AGENTS-Regeln setzen den Rahmen; eine kurze zentrale "Konfig-Konvention" in Nutzerdoku fehlt noch.
- [ ] Leichtgewichtige Code-Gates einfuehren (z. B. `ruff` + `pytest`) statt schwerem Tooling-Overkill.
  - Status (2026-02-25): `pytest`-Gate vorhanden; `ruff` noch offen.
- [ ] Kleine Architektur-Notizen pflegen, wenn wichtige Entscheidungen getroffen werden.
  - Status (2026-02-25): teils ueber Statusnotizen in ToDos erfasst, aber kein eigener schlanker ADR-/Entscheidungsbereich.

## 3) Nice-to-Have ToDos (wenn Zeit da ist)

- [ ] Einfachen CLI-Einstieg fuer zentrale Flows pruefen (optional).
- [ ] Mehr Typannotationen in stark genutzten Kernfunktionen.
- [ ] Erweiterte Benchmark-Szenarien fuer sehr grosse Datensaetze.

## 4) Praktischer Review-Ablauf (pro groesserer Aenderung)

Hinweis: Das ist eine wiederkehrende Arbeitscheckliste, kein einmalig global abhakbarer Block.

- [ ] Schritt 1: Funktioniert der End-to-end Smoke noch?
- [ ] Schritt 2: Sind Contract-Tests und relevante Unit-Tests gruen?
- [ ] Schritt 3: Sind Fehlermeldungen klar und fuer Forschende verstaendlich?
- [ ] Schritt 4: Ist die Aenderung im Notebook/README nachvollziehbar dokumentiert?
- [ ] Schritt 5: Wurde mindestens ein "In-1-Jahr-ich-verstehe-es-sofort"-Check gemacht (Namen, Struktur, Kommentare, Modularitaet)?

## 5) "In 1 Jahr noch vorbildlich" - Kurz-Kriterien

- [ ] Neue Kollegin kann in < 1 Tag einen kleinen Lauf reproduzieren.
- [ ] Kernmodule sind ohne tiefe Einarbeitung lesbar.
- [ ] Typische Fehler sind selbsterklaerend und schnell behebbar.
- [ ] Aenderungen an einem Modul brechen nicht still andere Pipeline-Teile.
- [ ] Performance ist fuer den realen Forschungs-Workflow gut genug und nachvollziehbar.
