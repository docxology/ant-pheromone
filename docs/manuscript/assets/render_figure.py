"""Plot the existing model predictions alongside the transcribed observations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pheromoneModel as model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    # The published entry point exposes predictions through its plot.
    model.runExp12_AnalyticalModel(model.gl_parameter_bundle_opt_12)
    original = plt.gcf()
    e_prediction, n_prediction = (
        line.get_data() for line in original.axes[1].get_lines()
    )
    plt.close(original)
    dynamic = model.simulateDynamic(model.gl_parameter_bundle_opt_4, plot_p=False)

    fig, (preference, switches) = plt.subplots(
        2, 1, figsize=(7.4, 5.7), layout="constrained"
    )
    for prediction, values, comparison, color, marker in [
        (e_prediction, model.gl_frac_table_e, "E+F vs E", "#A74729", "o"),
        (n_prediction, model.gl_frac_table_n, "E+F vs N", "#196185", "s"),
    ]:
        preference.plot(*prediction, color=color, label=f"{comparison} model")
        times = range(5, 5 * len(values) + 1, 5)
        preference.scatter(
            times, values, facecolors="none", edgecolors=color,
            marker=marker, s=20, label=f"{comparison} transcription",
        )

    switches.plot(dynamic, color="#176A58", label="Branch A model")
    switches.scatter(
        range(1, len(model.gl_frac_table_exp4)), model.gl_frac_table_exp4[1:],
        facecolors="none", edgecolors="#435463", s=15, label="Branch A transcription",
    )
    for minute in (model.gl_trial_duration, 2 * model.gl_trial_duration):
        switches.axvline(minute, color="#686868", linestyle="--", linewidth=1)
    preference.set(title="Experiments 1 and 2: trail decay", ylabel="E+F fraction")
    switches.set(title="Experiment 4: food switches", ylabel="Branch A fraction")
    for ax in (preference, switches):
        ax.set(xlabel="Minute", ylim=(0, 1))
        ax.grid(axis="y", color="#dddddd")
        ax.legend(loc="best", ncols=2, frameon=False, fontsize=8)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
