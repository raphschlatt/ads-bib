# ADS Pipeline (`ads-bib`) - Review ToDo (Realistic Scope)

Stand: 2026-02-20  
Ziel: Ein schlanker, aber gruendlicher Review-Plan fuer ein kleines Forschungsteam (kein Enterprise-Overhead).

Abgrenzung: `Package_ToDo.md` bleibt das Release/Packaging-Gate.  
Diese Datei ist fuer Codequalitaet, Stabilitaet und Wartbarkeit im Alltag.

## 0) Fuer wen ist das Package?

- [ ] Primaere Zielgruppe klar festhalten: Forschende/Promovierende, die ADS-Daten analysieren (Notebook-first, lokal, kleine Teams).
- [ ] Sekundaere Zielgruppe: technisch versierte Kolleg:innen, die einzelne Module als Python-Library nutzen.
- [ ] Nicht-Ziel dokumentieren: keine "always-on" SaaS-Plattform, kein 24/7-Betrieb, kein grosser Enterprise-MLOps-Stack.

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
- [ ] Logging vereinheitlichen: weniger unkontrollierte `print()`, stattdessen kontrollierbare Ausgabe (`verbose`/`quiet`).
- [ ] Public-Funktionen mit klaren Docstrings pflegen:
  - required columns
  - wichtige Parameter
  - Rueckgabeformat

### C) Tests, die wirklich helfen

- [x] Bestehende Contract-Tests gruen halten (Notebook + Schema).
  - Status (2026-02-25): Vollsuite inkl. Contract-Gates gruen (`146 passed`).
- [ ] Fuer jeden echten Bugfix mindestens einen Regressionstest schreiben.
- [x] Test-Suite in schnell/langsam aufteilen, damit lokales Feedback flott bleibt.
- [ ] Netz- und Modellabhaengige Teile weiterhin mocken, damit Tests stabil und reproduzierbar bleiben.

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

### E) Doku fuer reale Nutzer

- [ ] Eine klare "Happy Path"-Anleitung pflegen:
  - Umgebung
  - minimale Konfiguration
  - ein funktionierendes Beispiel
- [ ] "Troubleshooting"-Abschnitt fuer haeufige Probleme ergaenzen:
  - ADS-Token/API-Key
  - fehlende Extras
  - typische Fehlermeldungen
- [ ] Kurz dokumentieren, welche Teile stabil sind und wo noch Experimentierstatus gilt.

## 2) Should-Have ToDos (nach Must-Have)

- [ ] Public API bewusst markieren (was darf importiert werden, was ist intern).
- [ ] Einheitliche Konfig-Konvention fuer Notebook und Module festziehen.
- [ ] Leichtgewichtige Code-Gates einfuehren (z. B. `ruff` + `pytest`) statt schwerem Tooling-Overkill.
- [ ] Kleine Architektur-Notizen pflegen, wenn wichtige Entscheidungen getroffen werden.

## 3) Nice-to-Have ToDos (wenn Zeit da ist)

- [ ] Einfachen CLI-Einstieg fuer zentrale Flows pruefen (optional).
- [ ] Mehr Typannotationen in stark genutzten Kernfunktionen.
- [ ] Erweiterte Benchmark-Szenarien fuer sehr grosse Datensaetze.

## 4) Praktischer Review-Ablauf (pro groesserer Aenderung)

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
