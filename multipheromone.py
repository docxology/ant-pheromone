import matplotlib.pyplot as plt
import math

##################################
# 2021/10/18
# Experimenting with multi-component pheromones.
# The ant lays down a single pheromone during exploration or exploitation,
# but it contains two (for now) chemical components that evaporate at
# different rates.  The concentration ratio tells how long the pheromone
# has been there.
#
# Accurate measurement of a weak signal near the noise level really requires
# sampling and integrating the measurement over some period of time.
