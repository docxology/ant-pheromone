# Model and replay methods

The repository's Python module embeds observations transcribed from the
published trail-decay and dynamic plots [@ant-source; @dussutour2009].
For the decay experiments, `runExp12_AnalyticalModel()` applies exponential
decay to the starting signal above the model floor. It amplifies each
signal and maps the resulting pair to a branch-choice fraction. The
choice function uses distance from equal signal and overall signal
strength. When the two amplified signals are equal, it returns an even
split.

`simulateDynamic()` repeats the same branch-choice rule across successive
food-placement phases. At each minute the model allocates traffic across
branches according to the current preference. It decays the existing
trails and adds deposition according to each branch's traffic and food
state. The final concentrations from one phase become the inputs to
the next. The figure uses the separate fitted parameter bundles shipped
for the two experiment families [@ant-source].
