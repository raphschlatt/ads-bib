# Runs Directory

This repository tracks only the `runs/` folder structure.

Each local pipeline execution writes artifacts into a timestamped subfolder
under `runs/`. The project-wide cache belongs in `data/cache/`, not here.
Run artifacts are intentionally not committed to git.
