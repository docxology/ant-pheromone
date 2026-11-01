# AGENTS.md — ant-pheromone

Guidance for agents working in this repository.

## What this repository is

One Python module, `pheromoneModel.py` (about 1,085 lines), that tests a
single-pheromone hypothesis against the two-pheromone account of big-headed ant
recruitment. It is a published-model reproduction, not an application.

- Target paper: Dussutour, Nicolis, Shephard, Beekman & Sumpter (2009), "The role
  of multiple pheromones in food recruitment by ants", *J Exp Biol* 212:2337–2348,
  doi:10.1242/jeb.029827.
- This model's paper: Saund & Friedman (2023), "A Single-Pheromone Model Accounts
  for Empirical Patterns of Ant Colony Foraging Previously Modeled Using Two
  Pheromones", *Cognitive Systems Research* 80:81–89,
  doi:10.1016/j.cogsys.2023.02.005.

The claim under test: ants lay one pheromone whether exploring or exploiting, but
lay more while exploiting, leaving a stronger trail. Pheromone decays
exponentially with time. Two experimental conditions are compared, E+F vs E and
E+F vs N.

## Layout

| Path | Role |
|---|---|
| `pheromoneModel.py` | The whole model, simulation, and figure generation |
| `archives/` | Superseded implementations (`pheromone_simple_1.py`, `pheromones1.py`–`pheromones3.py`). Kept for provenance; not maintained |
| `docs/AGENTS.md` | Directory-scoped guidance (single-file research code, no packaging, no CI) |
| `README.md` | Purpose, citations, and run notes |

There is no `src/`–`tests/` split, no packaging metadata, and no CI. Do not add
them to "modernise" the repository: the module being directly importable is the
published artefact.

## Running

Dependencies: `matplotlib`, `numpy`, `scipy`.

```bash
python3 -m pip install matplotlib numpy scipy
python3 -c "import pheromoneModel as ph; ph.runExp12_LookupModel()"
python3 -c "import pheromoneModel as ph; ph.runExp12_AnalyticalModel(ph.gl_parameter_bundle_opt_12)"
python3 -c "import pheromoneModel as ph; ph.simulateDynamic(ph.gl_parameter_bundle_opt_4)"
```

The module header lists every entry point, including the parameter-optimisation
functions. `gl_parameter_bundle_opt_12` and `gl_parameter_bundle_opt_4` are the
fitted bundles for the two experiments; `gl_parameter_bounds`,
`parameterBundleToArray`, and `parameterBundleArrayToDict` are the optimisation
plumbing.

## Conventions

- Module-level `gl_*` names are deliberate: they are the model's global
  parameter and lookup-table state, and the entry points mutate them. Preserve
  that style rather than threading state through arguments; changing it would
  break the reproduction.
- `archives/` is append-only history. Do not edit or delete files there.
- Numerical changes alter published results. If you change a constant or a
  solver, say which figure or table it moves, and regenerate that output.

## Verification

There is no test suite. Verification means regenerating the figures for both
experimental conditions and comparing them with the published figures. Report
which entry point you ran and which figure you inspected; do not describe the
model as passing tests it does not have.
