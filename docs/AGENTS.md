# AGENTS.md — ant-pheromone docs/

## Scope
Agent guidance for this documentation directory.

## Layout
- `README.md` — human entry point (repo purpose, model context, run notes)
- `manuscript/MANUSCRIPT_STATUS.md` — why no template manuscript applies here

## Conventions observed in this repo
- Single-file research code (`pheromoneModel.py`); no `src/`–`tests/` split, no
  packaging metadata, no CI.
- `archives/` holds archived material; `.idea/` is editor state; `__pycache__/`
  is ignored build output.

## Maintenance
Keep `README.md` aligned with the paper's status as the owner updates it: the
model paper is published (Saund & Friedman, Cognitive Systems Research
2023;80:81-89, doi:10.1016/j.cogsys.2023.02.005). Take run commands and
parameter names from the `pheromoneModel.py` header; do not invent them.

## Manuscript status
See `docs/manuscript/MANUSCRIPT_STATUS.md`.
