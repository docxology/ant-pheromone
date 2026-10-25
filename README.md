The repository, ant-pheromone, contains the program, pheromoneModel.py.
This program demonstrates that a single pheromone model accounts for certain experimental observations of foraging behavior in big headed ants.

The experimental paper and original two pheromone model is:
"The role of multiple pheromones in food recruitment by ants"
A. Dussutour, S. C. Nicolis, G. Shephard, M. Beekman, and D. J. T. Sumpter
J Exp Biol. 2009;212: 2337–2348. doi:10.1242/jeb.029827

Our paper offering a one pheromone model is:
"A Single-Pheromone Model Accounts for Empirical Patterns of Ant Colony Foraging Previously Modeled Using Two Pheromones"
E. Saund and D. A. Friedman
Cognitive Systems Research 2023;80: 81-89. doi:10.1016/j.cogsys.2023.02.005

## Run

The program runs in Python 3 with matplotlib, numpy, and scipy:

    python3 -m pip install matplotlib numpy scipy

From the repository directory:

    >>> import pheromoneModel as ph
    >>> ph.runExp12_LookupModel()
    >>> ph.runExp12_AnalyticalModel(ph.gl_parameter_bundle_opt_12)
    >>> ph.simulateDynamic(ph.gl_parameter_bundle_opt_4)

The module header of `pheromoneModel.py` lists all entry points, including the
parameter-optimization functions. There is no test suite; the entry points
display their results as plots.

