# Ant foraging model documentation

- [Setup](#setup)
- [Run the model](#run-the-model)
- [Verify the curves](#verify-the-curves)
- [Reproduction manuscript](manuscript/README.md)

## Setup

The reproduction uses [a single Python module](../pheromoneModel.py). Install
[uv](https://docs.astral.sh/uv/) and prepare the pinned Python 3.12 dependency
environment from the repository root.

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python docs/requirements.txt
```

The older implementations under [archives](../archives/) remain a historical
record. The run instructions use the module at the repository root.

## Run the model

```bash
.venv/bin/python -c 'import pheromoneModel as ph; ph.runExp12_LookupModel()'
.venv/bin/python -c 'import pheromoneModel as ph; ph.runExp12_AnalyticalModel(ph.gl_parameter_bundle_opt_12)'
.venv/bin/python -c 'import pheromoneModel as ph; ph.simulateDynamic(ph.gl_parameter_bundle_opt_4)'
```

Each invocation displays a Matplotlib plot when a graphical backend is
available. The lookup plot uses nearest neighbors from the transcribed
preference tables. The analytical plot uses the fitted bundle for the
trail-decay comparisons. The dynamic model uses the fitted bundle for
the food-switch experiment, and `simulateDynamic(..., plot_p=False)`
returns its branch-A trajectory for analysis.

Use the default inputs for the lookup reproduction. The current function
uses a lookup table and decay rate initialized at import, even when the
caller supplies different starting concentrations. The analytical function
accepts a parameter bundle and computes its own decay rate. The three
`findOptimalParameters_*` functions rerun the original optimizations.
The routine reproduction uses the stored fitted bundles.

## Verify the curves

Generate a headless figure showing the predictions and transcribed
observations. This command writes only to the path you supply.

```bash
.venv/bin/python docs/manuscript/assets/render_figure.py /tmp/ant-foraging-curves.png
```

Check both decay comparisons, the food-switch timing, and the plotted
fraction range. The [manuscript](manuscript/README.md) explains the origin
and limitations of the embedded observations. The project ships no unit-test
suite. Its native verification exercises the original plot functions and
checks the resulting curves.
