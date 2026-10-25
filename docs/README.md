# ant-pheromone — Documentation

`ant-pheromone` contains one program, `pheromoneModel.py`, demonstrating that a
single pheromone model accounts for experimental observations of foraging
behavior in big headed ants.

## Background (from the repo README)

- Empirical paper modeled: "The role of multiple pheromones in food recruitment
  by ants" — Dussutour, Nicolis, Shephard, Beekman & Sumpter, J Exp Biol.
  2009;212:2337–2348, doi:10.1242/jeb.029827
- This repo's paper: "A Single-Pheromone Model Accounts for Empirical Patterns
  of Ant Colony Foraging Previously Modeled Using Two Pheromones" — Saund &
  Friedman, Cognitive Systems Research 2023;80:81-89,
  doi:10.1016/j.cogsys.2023.02.005

## Run / test

The program runs in Python 3 with `matplotlib`, `numpy`, and `scipy`
(`python3 -m pip install matplotlib numpy scipy`; also noted in README.md).
There is no packaging metadata and no test suite. Import the module from the
repository directory and call an entry point, which displays its result as a
plot:

```python
>>> import pheromoneModel as ph
>>> ph.runExp12_LookupModel()
>>> ph.runExp12_AnalyticalModel(ph.gl_parameter_bundle_opt_12)
>>> ph.simulateDynamic(ph.gl_parameter_bundle_opt_4)
```

The module header of `pheromoneModel.py` documents every entry point and the
available parameter bundles; the parameter-optimization functions
(`findOptimalParameters_Exp12_AnalyticalModel`, `findOptimalParameters_Exp4`,
`findOptimalParameters_Exp12_Exp4`) are long-running, so they are not part of a
routine smoke check.

## Layout

- `pheromoneModel.py` — the model
- `archives/` — archived earlier iterations (`pheromone_simple_1.py`,
  `pheromones1.py`, `pheromones2.py`, `pheromones3.py`), kept for provenance
- `README.md` — background and citation context

## Status

Docs hub created by the docs-audit pass (2026-08-29); run instructions updated
2026-10-25.
