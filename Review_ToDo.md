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

- [ ] Grosses Modul `src/ads_bib/topic_model.py` schrittweise aufteilen (nur entlang klarer Fachgrenzen, keine Mega-Refactors auf einmal).
- [ ] Tote Pfade entfernen: unused Imports, ungenutzte Parameter, alte Alias-Namen.
- [ ] Logging vereinheitlichen: weniger unkontrollierte `print()`, stattdessen kontrollierbare Ausgabe (`verbose`/`quiet`).
- [ ] Public-Funktionen mit klaren Docstrings pflegen:
  - required columns
  - wichtige Parameter
  - Rueckgabeformat

### C) Tests, die wirklich helfen

- [ ] Bestehende Contract-Tests gruen halten (Notebook + Schema).
- [ ] Fuer jeden echten Bugfix mindestens einen Regressionstest schreiben.
- [ ] Test-Suite in schnell/langsam aufteilen, damit lokales Feedback flott bleibt.
- [ ] Netz- und Modellabhaengige Teile weiterhin mocken, damit Tests stabil und reproduzierbar bleiben.

### D) Performance mit realistischem Anspruch

- [ ] Kleine Baseline messen (z. B. 1k und 10k Dokumente): Laufzeit pro Hauptschritt + grober RAM-Footprint. Ist es RAM sparsam auch für große Datasets?
- [ ] Caching-Verhalten pruefen: zweiter Lauf muss sichtbar schneller sein, ohne falsche Wiederverwendung.
- [ ] Nur datenbasierte Optimierungen umsetzen (keine "vorsorglichen" Mikro-Optimierungen).

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
