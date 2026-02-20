# ADS Pipeline — Comprehensive Code Review ToDo

Stand: 2026-02-20
Ziel: Alles, was dieses Repo vorbildlich, stabil, lesbar, wartbar und professionell macht — unabhaengig von Packaging/Release (dafuer siehe `Package_ToDo.md`) und unabhaengig vom allgemeinen Review-Rahmen (siehe `Review_ToDo.md`).

Hier stehen die konkreten, zeilenspezifischen Befunde aus einem systematischen Code-Review.

---

## 1) Type Hints vervollstaendigen

- [x] **Private Helper in `topic_model.py` annotieren**
  - `_embed_local()` L152, `_embed_huggingface_api()` L161, `_reduce()` L297
  - `_cost_cb()` L526, `_wrapped()` L735, `filter()` L920
  - `TrackedOpenAINamer.__init__()` L595, `_record_usage()` L599, `_call_llm()` L612, `_call_llm_with_system_prompt()` L623
- [x] **`export.py:48` — `_export_chunk()` hat keine Type Hints**
- [x] **`visualize.py` — `WordCloud`-Properties `javascript`, `html`, `css` annotieren**
- [x] **`_utils/costs.py` — `__init__()` L9, `summary()` L41 annotieren**
- [x] **`run_manager.py:6` — `from typing import Any, Dict` auf `dict[str, Any]` umstellen**
  - Rest des Codebases nutzt lowercase generics konsistent.

---

## 2) Docstrings ergaenzen

Stil: NumPy-Format (wird bereits konsistent fuer Public API verwendet).

- [x] **`citations.py` — 8 Private Helper ohne Docstrings**
  - `_author_list()`, `_author_text()`, `_first_author_lastname()`, `_has_value()`
  - `_filter_by_authors()`, `_format_author()`, `_format_ref_author()`, `_format_pub()` (43 Zeilen, nicht trivial)
- [x] **`topic_model.py` — 16 Funktionen ohne Docstrings**
  - Insb. `_embed_local()`, `_embed_huggingface_api()`, `_embed_openrouter()`, `_reduce()`, `_record_llm_usage()`, `_build_representation_model()`, `_create_llm()`
- [x] **`visualize.py` — `WordCloud.__init__()` und Property-Contracts dokumentieren**
- [x] **`_utils/costs.py` — `CostTracker`-Properties und `__repr__()` dokumentieren**
- [x] **`_utils/openrouter_costs.py` — `_get_mapping_value()`, `_coerce_float()`, `_coerce_int()` dokumentieren**

---

## 3) Error Handling verbessern

- [x] **Bare `except Exception:` mit Logging ersetzen**
  - [x] `translate.py:263` — Uebersetzungsfehler werden still geschluckt, kein Hinweis *welche* Eintraege und *warum* fehlgeschlagen sind.
  - [x] `translate.py:290` — Gleich, HuggingFace-Pfad.
  - [x] `topic_model.py:172` — Retry in `_embed_huggingface_api()` loggt nicht bei Fehlschlag.
  - [x] `topic_model.py:219` — Retry in `_embed_openrouter._embed_batch()` ebenso.
  - [x] `_utils/openrouter_costs.py:153` — `fetch_generation_cost()` ignoriert Netzwerkfehler still.
- [x] **`export.py:87` — Outer except in `_export_chunk()` loggt nicht, *was* fehlgeschlagen ist**
- [x] **`visualize.py:428` — Warning als `print()` statt `warnings.warn()`**
  - Fallback (cluster boundaries deaktiviert) ist signifikant, sollte klarer kommuniziert werden.

---

## 4) Dead Code und Imports aufraeumen

- [x] **`citations.py:16` — `import os` entfernen** (wird nirgends verwendet, nur `pathlib.Path`)
- [x] **`run_manager.py:1` — `import os` entfernen** (wird nirgends verwendet, nur `pathlib.Path`)
- [x] **`export.py:90` — doppelter `import time as _t` entfernen** (`time` ist bereits in L51 importiert)
- [x] **Inline-Imports auf Modul-Ebene verschieben**
  - `export.py:51` — `import time`
  - `topic_model.py:175` — `import time` innerhalb einer Schleife

---

## 5) Code-Komplexitaet reduzieren

### Lange Funktionen aufbrechen

- [x] **`translate.py:translate_dataframe()` — 166 Zeilen, Nesting-Tiefe 6**
  - Provider-spezifische Logik in eigene Subfunktionen extrahieren.
- [x] **`topic_model.py:fit_toponymy()` — 158 Zeilen**
  - Setup, Fit, Post-Processing als benannte Schritte separieren.
- [x] **`visualize.py:create_topic_map()` — 174 Zeilen**
  - Tooltip-Aufbau, Label-Vorbereitung, Plot-Konfiguration in Helper auslagern.
- [x] **`_utils/openrouter_costs.py:summarize_openrouter_costs()` — 122 Zeilen**
  - Fetch-Phase und Aggregation-Phase trennen.

### Duplikate beseitigen (DRY)

- [x] **Retry-Logik konsolidieren**
  - `export.py:_export_chunk()` L55–93 reimplementiert `retry_request()` aus `_utils/ads_api.py` nahezu identisch.
  - `topic_model.py:_embed_huggingface_api()` L167–176 und `_embed_openrouter._embed_batch()` L202–223 haben eigene Retry-Loops.
  - Ziel: Alle 4 Stellen sollen `retry_request()` (oder eine generalisierte Variante) nutzen.
- [x] **OpenRouter API-Base-URL: Konstante nur einmal definieren**
  - `_utils/openrouter_costs.py:9` definiert `DEFAULT_OPENROUTER_API_BASE`.
  - `topic_model.py:761` dupliziert den Wert als Default-Argument.
- [x] **Author-Serialisierung vereinheitlichen**
  - `citations.py:62-63` (`_author_text()`) und `visualize.py:328` (Inline-Lambda) machen dasselbe.

---

## 6) Hardcoded Values konfigurierbar machen

- [x] **`search.py:47` — ADS Search-URL als benannte Konstante**
  - `export.py` hat bereits `ADS_EXPORT_URL`; `search.py` hat die URL inline.
- [x] **`search.py:76` — Link-Gateway-URL als benannte Konstante**
- [x] **`topic_model.py:953, 971` — `"en_core_web_sm"` hardcoded**
  - Ist ein anderes spaCy-Modell als `en_core_web_lg` in `tokenize.py`; sollte Parameter sein.
- [x] **Magic Numbers dokumentieren oder als benannte Konstanten**
  - `topic_model.py:307` — `random_state=42` in `_reduce()` ist nicht uebersteuerbar
  - `topic_model.py:314` — `n_neighbors=80` (UMAP) vs. L307 `n_neighbors=60` (PaCMAP) — undokumentiert
  - `topic_model.py:345, 357` — `min_cluster_size=180` Default
  - `topic_model.py:492` — `top_n_words=20` nicht exponiert
  - `translate.py:136` — `max_tokens=2048` nicht exponiert
- [x] **CDN-URLs in `visualize.py` als Konstanten**
  - L46 jQuery, L74 d3-cloud — pinned Versionen in Inline-Strings.

---

## 7) Naming-Inkonsistenzen

- [x] **`citations.py:137` — Spalte `"count"` speichert den Referenz-*Index*, nicht einen Count**
  - Umbenennen zu `"ref_index"` oder `"ref_position"`.
- [x] **`run_manager.py` — Emoji in `print()`-Aufrufen entfernen**
  - L46, L47, L93 — inkonsistent mit dem Rest des Codebases.

---

## 8) Logging-Strategie

- [x] **Entscheidung treffen: `print()` vs. `logging`**
  - Aktuell: 64x `print()` in 10 Dateien, `logging` nur fuer BERTopic-Warning-Suppression.
  - Fuer Notebook-Nutzung ist `print()` akzeptabel, aber:
    - Debug-/Diagnostic-Ausgaben sollten abschaltbar sein.
    - Warnings (z.B. `visualize.py:428`) sollten `warnings.warn()` nutzen.
  - Empfehlung: `print()` fuer User-facing Output beibehalten, `warnings.warn()` fuer Warnungen, optional `logging` fuer Debug.
  - **Entscheidung umgesetzt:** Notebook-first bleibt `print()` fuer Fortschritt/Status; nicht-fatales Verhalten nutzt `warnings.warn()`; kein globales Logging-Framework fuer normale Pipeline-Ausgabe.
- [x] **Alle `print("Warning: ...")` Stellen auf `warnings.warn()` umstellen**

---

## 9) Test-Abdeckung erweitern

### Komplett ungetestet (Prioritaet nach Risiko)

- [x] **`search.py`** — `search_ads()`, `save_search_results()` — API-Interaktion, kritisch
- [x] **`translate.py`** — `detect_languages()`, `translate_dataframe()` — komplexe Logik, tiefes Nesting
- [x] **`tokenize.py`** — `tokenize_texts()` — 65 Zeilen, spaCy-Abhaengigkeit
- [x] **`curate.py`** — `get_cluster_summary()`, `remove_clusters()`, `filter_by_field()` — Datenfilterung
- [x] **`_utils/cleaning.py`** — `clean_html()`, `clean_range()`, `clean_dataframe()` — Parser-Logik
- [x] **`_utils/io.py`** — alle 6 I/O-Funktionen — Serialisierung ist fehleranfaellig
- [x] **`config.py`** — `init_paths()`, `load_env()`
- [x] **`run_manager.py`** — `RunManager`, `save_config()`, `get_path()`

### Teilweise ungetestet

- [x] **`citations.py`** — `create_co_citations()`, `create_bibliographic_coupling()`, `export_wos_format()`, `process_all_citations()`
- [x] **`topic_model.py`** — `fit_bertopic()`, `reduce_dimensions()`, `cluster_documents()` (nur Toponymy-Pfad ist getestet)
- [x] **`visualize.py`** — `create_topic_map()` — keine Tests

### Test-Infrastruktur

- [x] **`test_pipeline_notebook_contract.py:7` — relativer Pfad `Path("pipeline.ipynb")`**
  - Test funktioniert nur wenn `pytest` vom Projekt-Root ausgefuehrt wird.
  - Robust machen mit `Path(__file__).parent.parent / "pipeline.ipynb"`.

---

## 10) Notebook-Qualitaet

- [ ] **`pipeline.ipynb` — Stale Outputs pruefen**
  - Keine veralteten Schema-Namen oder historischen Logs in gespeicherten Cell-Outputs.
- [ ] **Cell-Dokumentation pruefen**
  - Jede Phase sollte einen kurzen Markdown-Header haben, der erklaert *was* und *warum*.
- [ ] **Reproduzierbarkeit sicherstellen**
  - Alle `random_state`-Werte explizit gesetzt?
  - Keine impliziten Abhaengigkeiten von vorherigen Runs?

---

## Zusammenfassung nach Prioritaet

| Prioritaet | Kategorie | Aufwand |
|------------|-----------|---------|
| Hoch | Test-Abdeckung erweitern (9) | Gross |
| Hoch | Error Handling verbessern (3) | Mittel |
| Hoch | Retry-Logik konsolidieren (5/DRY) | Mittel |
| Mittel | Dead Code/Imports aufraeumen (4) | Klein |
| Mittel | Lange Funktionen aufbrechen (5) | Mittel |
| Mittel | Hardcoded Values (6) | Klein–Mittel |
| Mittel | Type Hints vervollstaendigen (1) | Mittel |
| Mittel | Docstrings ergaenzen (2) | Mittel |
| Niedrig | Naming-Inkonsistenzen (7) | Klein |
| Niedrig | Logging-Strategie (8) | Klein |
| Niedrig | Notebook-Qualitaet (10) | Klein |
