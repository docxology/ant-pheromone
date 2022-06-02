
#pheromones2.py is for testing alternative hypotheses to the two-pheromnes
#claim put forth in:
#"The role of multiple pheromones in food recruitment by ants"
#A. Dussutour, S. C. Nicolis, G. Shephard, M. Beekman, and D. J. T. Sumpter
#J Exp Biol. 2009;212: 2337–2348. doi:10.1242/jeb.029827

#Let us  assume ants put down the same kind of pheromone whether exploring or
#exploiting / gathering.  But they put down more when exploiting, to leave a
#stronger trail for where to go.  We'll assume that pheromone decays exponentially
#with time.
#
#There are two experimental conditions, the E+F vs E condition, and the E+F vs N condition.
#The E+F branch has pheromone after the ants have both explored and exploited.
#The E branch has pheromone after the ants have only explored, but not exploited.
#The N branch has no pheromone.
#
#Our hypothesis is that the E+F branch leaves a lot of pheromone, say, 10, and
#the E branch leaves a small but significant amount of pheromone, say, 1.
#So exploiting leaves 10 times as much pheromone as just exploring.
#
#When there are good amounts of pheromone, then the measurement of
#which path has more pheromone is clear.  Preferably choose the path with more.
#When there is skimpy amount of pheromone, then hedge bets by allocating
#traversals to both paths.
#
#We decompose the computation into two stages, signal amplification, and
#path preference.
#
#This file contains two top level functions that produce branch preference
#curves.  One function is for Dussutour Experiments 1 & 2, and another uses
#the same model to simulate Dussutour Experiment 4, "dynamic environment."
#

###########################
#
#How to run:
#

#This runs in Python 3


# Install/import any dependencies
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import math

#>>> import pheromones1 as ph


#Plot fraction of E+F branch as a function of time using the lookup table data
#obtained from Fig. 2.   This is not a parametric model, but it is a preliminary
#test showing that branch preference can be indexed by a single pheromone amount
#parameter.
#>>> ph.runExp12_LookupModel()



#Plot the time course of hypothesized physical pheromone, amplifed pheromone
#measurement, and branch preference, based on the amplification + sigmoid
#product analytical model.
#>>> ph.runExp12_AnalyticalModel()

#Plot branch preference over time given two switches in the branch leading
#to food, using the analytical model.  This predicts the preference curves
#produced by Dussutour et al. Experiment 4.
#>>> ph.simulateDynamic()

#Set parameters for simulateDynamic() to the same parameters used for
#the Experiment 1 & 2 analytical model.
#>>> ph.setDyanmicSimulationParameters('exp2')

#Set parameters for simulateDynamic() to values that better match the
#preference trajectory of Dussutor's Experiment 4.
#>>> ph.setDyanmicSimulationParameters('exp4')

#
#
###########################







# The preference table tells the proportion of ants choosing the E+F branch in
# two conditions, E+F vs. N (experiment 1), and E+F vs. E (experiment 2).
# These preference values are the curves in Fig. 2
# each increment is 5 time steps.
# time axis           5    10   15   20   25   30   35   40   45   50
#                   55    60   65   70   75   80   85   90   95

# Experiment 1 behavioral results 
# fraction of ants choosing the E+F branch vs. the N branch.
gl_frac_table_n = [.94, .93, .89, .81, .78, .78, .80, .78, .76, .77,
                   .73, .75, .74, .70, .66, .65, .54, .54, .48]

# Experiment 2 behavioral results
# fraction of ants choosing the E+F branch vs. the E branch
gl_frac_table_e = [.94, .91, .83, .75, .68, .62, .56, .51, .52, .49,
                   .50, .50, .50, .49, .51, .50, .50, .50, .51]

# These empirical behavioral preferences are converted to a lookup table
# that tells branch preference ratio as a function of amount of pheromone on each branch. 
# The conversion assumes an exponential decay in the amount of pheromone.

# c1 is the assumed amount of pheromone on the E+F branch at the start
# c2 is the assumed amount of pheromone on the E branch at the start
# Decay of c1 to the noise level seems to occur after 90 steps.
# Call the noise level 0.02.
# this gives a decay rate of -.069
# 10 exp(90 * -.069) = 0.02

gl_decay = -.069
# Estimated from the data, how long it takes
# E+F vs. N to reach the pheromone noise level,
# starting from c1 = 10.

gl_noise_level = .02
# minimum signal level, initial estimate of 0.02


def setupTable(c1=10, c2=1):
    lut_ph1_ph2_frac = []  # (ph1, ph2, frac)
    c1 = c1 - gl_noise_level
    c2 = c2 - gl_noise_level
    for i in range(0, 19):
        frac_e = gl_frac_table_e[i]
        frac_n = gl_frac_table_n[i]
        t = (i+1)*5  # In Fig. 2, measurements start after 5 minutes
        ph1 = c1 * math.exp(gl_decay * t) + gl_noise_level
        ph2 = c2 * math.exp(gl_decay * t) + gl_noise_level
        print('t: ' + str(t) + ' frac_e: ' + str(frac_e) + ' frac_n: '
              + str(frac_n) + ' ph1: ' + str(ph1) + ' ph2: ' + str(ph2))
        lut_ph1_ph2_frac.append((ph1, ph2, frac_e))
        lut_ph1_ph2_frac.append((ph1, gl_noise_level, frac_n))
    return lut_ph1_ph2_frac


def setTable():
    global gl_lut
    gl_lut = setupTable()


# gl_lut is a list of tuples:
# (pheromone1_level, pheromone2_level, preference_fraction)

try:
    gl_lut
except NameError: 
    gl_lut = setupTable()


def getLUT():
    return gl_lut


# plot fraction of E+F branch as a function of time using the lookup table data
# obtained from Fig. 2
# This assumes a single pheromone starting with concentrations 10, 1, and .02
# for the E+F, E, and N branches, respectively.
# This is not a parametric model.  This is directly using the data entered into
# the lookup table simply to re-compute the plots shown in the data.
# This just demonstrates that a single pheromone decay model can recapitulate
# the patterns of Fig. 2. Thus, this claim in the paper is overstated:
# "For example, the observation that the initial frequency of ants choosing
#  the E+F branch is the same in the E+F vs N and E+F vs E experiments but
#  these frequencies diverge after 15 min (Fig. 2) is difficult to explain
#  with a single pheromone." (a single pheromone model isn't tested) .
# In fact, a single pheromone model does the job.
#(This was formerly called testExp2 in an earlier iteration of the program.)
def runExp12_LookupModel(c1=10, c2=1):
    duration = 120
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    br1_ar = []   # exponential decaying pheromone on branch 1
    br2a_ar = []  # exponential decaying pheromone on branch 2
    # fbr1_ar = [] # assigned but never used.
    # fbr2a_ar = [] # assigned but never used.
    # fbr2b_ar = [] # assigned but never used.
    ratio2a_ar = []
    ratio2b_ar = []
    for m in range(duration):   # duration is in minutes
        br1 = c1 * math.exp(gl_decay * m) + gl_noise_level
        br2a = c2 * math.exp(gl_decay * m) + gl_noise_level
        br2b = gl_noise_level
        e_condition_frac = lookupFraction(br1, br2a, gl_lut)
        n_condition_frac = lookupFraction(br1, br2b, gl_lut)
        br1_ar.append(br1)
        br2a_ar.append(br2a)
        ratio2a_ar.append(e_condition_frac)
        ratio2b_ar.append(n_condition_frac)
    ax1.plot(br1_ar)
    ax1.plot(br2a_ar)
    ax2.plot(ratio2a_ar, color="r")
    ax2.plot(ratio2b_ar, color="b")
    plt.xlim([0, duration])
    plt.show()


# find the entry in the lookup table closest to the ph1/ph2 fraction provided
# Since the data set is so small, do a linear search to find the closest match.
def lookupFraction(ph1, ph2, lut, print_p=False):
    nearest_dist = 1000000
    nearest_frac = -1
    if print_p:
        print('\nph1: ' + str(ph1) + ' ph2: ' + str(ph2))
    for item in lut:
        dist2 = pow((ph1 - item[0]), 2) + pow((ph2 - item[1]), 2)
        if dist2 < nearest_dist:
            if print_p:
                print('  dist2: ' + str(dist2) + ' item: ' + str(item))
            nearest_dist = dist2
            nearest_frac = item[2]
    if print_p:
        print(' ans: ' + str(nearest_frac))
    return nearest_frac


# Plot Lookup table of preference fraction vs  measured pheromone on 2 branches
# The measured pheremone is the exponentially decreasing amount of pheromone,
# but mapped through the measurement function amplifySignal()
# which nonlinearly amplifies small amounts.
# Preference fraction is the size of the circle.
def plotLUT(max_range=10):
    global gl_lut
    plt.figure(figsize=(6, 6))
    # min_v = 0 # assigned but not used
    # max_v = 10 # assigned but not used
    x_ar = []
    y_ar = []
    val_ar = []
    for item in gl_lut:
        area = item[2] - .4
        val_ar.append(area*area*400)
        # print(str(area))
        x = item[0]
        y = item[1]
        x_amp = amplifySignal(x)
        y_amp = amplifySignal(y)
        # x_amp = x  #when not commented out, plot the raw amounts
        # y_amp = y
        x_ar.append(x_amp)
        y_ar.append(y_amp)
    plt.scatter(x_ar, y_ar, val_ar)
    plt.xlim([-.1, max_range])
    plt.ylim([-.1, max_range])
    plt.show()


# These parameters were estimated by plotting the observed data from Fig. 2
# and adjusting the parameters until the preference ratio transition occurred
# at about the same diagonal for E+F vs. E condition and E+F vs. N condition.
def amplifySignal(x):
    xpr = x - gl_noise_level
    val = 5.5*pow(xpr, .25)
    return val


# Returns distance to the midline of equal pheromone amounts.
# As everywhere, units are arbitrary pheromone amounts, assuming the
# E+F branch starts with pheromone amount 10.
def distToMid(xpr, ypr):
    xm = (xpr + ypr)/2
    ym = xm
    dist = pow(pow(xpr-xm, 2) + pow((ypr-ym), 2), .5)
    return dist


# These parameters were estimated from plot of the observed data from Fig. 2.
# This squashing function creates a band of equal preference near the midline,
# transitioning to a preference for one branch or the other at a distance of
# about 2 from the midline.
def fracFromDistToMid(dist):
    frac = 1.0 / (1 + math.exp(-4*(dist-2))) - 1 / (1 + math.exp(8))    # ch1
    return frac


# The ratio map is a function of the distance to the midline fbr1 = fbr2
# and also the distance to the origin fbr1 = fbr2 = 0.
# There is a radius of about 8.
# Above this radius, the ratio can go up to 1.0
# Below this radius, the ratio is limited to about 0.6.
def mapGetRatio(fbr1, fbr2):
    dist_to_mid_2 = distToMid(fbr1, fbr2)
    dist_to_mid_ratio_factor = fracFromDistToMid(dist_to_mid_2)
    dist_to_orig = pow(pow(fbr1, 2) + pow((fbr2), 2), .5)
    dist_to_orig_ratio_factor = .6 + .4 * 1 / (1 + math.exp(-.8 * (dist_to_orig-8)))
    total_ratio = (dist_to_mid_ratio_factor * dist_to_orig_ratio_factor) / 2 + .5
    if fbr2 > fbr1:
        total_ratio = 1.0 - total_ratio
    return total_ratio


# Top level function.
# plot fraction of E+F branch as a function of time using a model of
# signal detection/amplification and then relative amounts of these,
# where the model form and parameters have been estimated in stages from Fig. 2
# The model is:
#  1. Amplify pheromone to get a measurement signal
#  2. Convert the measurement signals from the two branches to preference ratio
#     2a. Find distance from the signal amounts to the midpoint of the
#         equal-amounts line in measurement space. Map using squashing function
#     2b. Find the distance from the signal amounts to the origin.
#         Map using a squashing function.
#     2c. Multiply the two considerations, distance to midpoint and to origin.
# This assumes a single pheromone starting with concentrations 10, 1, and .02
# for the E+F, E, and N branches, respectively.
#(This was formerly called testExp4 in an earlier iteration of the program.)
def runExp12_AnalyticalModel(c1=10, c2=1):
    c1 = c1 - gl_noise_level
    c2 = c2 - gl_noise_level
    duration = 120
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    br1_ar = []   # exponential decaying pheromone on branch 1
    br2a_ar = []  # exponential decaying pheromone on branch 2
    fbr1_ar = []
    fbr2a_ar = []
    fbr2b_ar = []
    ratio2a_ar = []
    ratio2b_ar = []
    # dist_a_ar = [] # assigned but not used
    # dist_b_ar = [] # assigned but not used
    for m in range(duration):   # duration in minutes
        br1 = c1 * math.exp(gl_decay * m) + gl_noise_level
        br2a = c2 * math.exp(gl_decay * m) + gl_noise_level
        br2b = gl_noise_level
        fbr1 = amplifySignal(br1)
        fbr2a = amplifySignal(br2a)
        fbr2b = amplifySignal(br2b)
        br1_ar.append(br1)
        br2a_ar.append(br2a)
        fbr1_ar.append(fbr1)
        fbr2a_ar.append(fbr2a)
        fbr2b_ar.append(fbr2b)
        ratio_2a = mapGetRatio(fbr1, fbr2a)
        ratio_2b = mapGetRatio(fbr1, fbr2b)
        ratio2a_ar.append(ratio_2a)
        ratio2b_ar.append(ratio_2b)
    ax1.plot(br1_ar)
    ax1.plot(br2a_ar)
    ax1.plot(fbr1_ar, "g")
    ax1.plot(fbr2a_ar, "m")
    ax1.plot(fbr2b_ar, "m")
    ax2.plot(ratio2a_ar, color="r")
    ax2.plot(ratio2b_ar, color="b")
    plt.xlim([0, duration])
    plt.show()



    
##################################

# Simulating Experiment 4, "dynamic environment" alternating food placement.

# We can use with the parameters used for Experiment 1-2, but the plot has
# some differences in appearance from Fig. 10.
# We can adjust the parameters to better fit Fig. 10, but then there is not
# as close a fit to Fig. 2.  Really, we would need more experimental data to
# find the right functional form and parameters to fit both figures well and
# predict other experimental results about ant behavior in this kind of setup.


# gl_ants_per_m = 50   #ch2
gl_ants_per_m = 30    #better fit to Exp. 4 data



gl_deposit_rate_exploit = .011    # pheromone per ant deposited
gl_deposit_rate_explore = .0011   # pheromone per ant deposited


gl_trial_duration = 45

# To simulate their experiment of placing food at one of the platforms at
# the end of a branch and watching activity ramp up and down, we need to
# model pheromone deposition as a function of whether the ant is exploring
# or exploiting.
# c1 and c2 are initial pheromone levels at the left (1) and right (2) branch
# respectively.
# f1 & f2 are whether food is present at the left or right branch, respectively


#Set to True to use the parameters from Experiments 1 and 2 for the
#simulation of Experiment 4.
gl_exp_2_parameters_p = False

#call with directive = 'exp2' or 'exp4'
def setDynamicSimulationParameters(directive = None):
    if directive != 'exp2' and directive != 'exp4':
        print('To set Exp4 simulation parameters from Dussutour Exp. 1&2 data: >>> ph.setExp4Parameters(\'exp2\').\n To set Exp4 simulation parameters from Dussutour Exp. 4 data: >>> ph.setExp4Parameters(\'exp4\').')
        return
    global gl_exp_2_parameters_p
    global gl_ants_per_m
    if directive == 'exp2':
        gl_exp_2_parameters_p = True
        gl_ants_per_m = 50    #as described in the Dussutour paper
    elif directive == 'exp4':
        gl_exp_2_parameters_p = False
        gl_ants_per_m = 30    #a better fit to the Exp. 4 data
    else:
        print('unrecognized directive: ' + str(directive))



def simulateYJunctionTravel(c1=0, c2=0, f1=False, f2=False, ratio_ar=None):
    duration = gl_trial_duration
    ph1 = max(c1, gl_noise_level)
    ph2 = max(c2, gl_noise_level)
    print(f1)
    if ratio_ar is None:
        ratio_ar = []
    for m in range(duration):   # duration in minutes

        #choose whether to use paremeters chosen to fit Dussutour Experiments 1 and 2,
        #or else use parameters that better fit their Experiment 4.
        if gl_exp_2_parameters_p:
            fph1 = amplifySignal(ph1)
            fph2 = amplifySignal(ph2)                    
            frac_1 = mapGetRatio(fph1, fph2) 
        else:
            fph1 = amplifySignal_Exp4(ph1)
            fph2 = amplifySignal_Exp4(ph2)
            frac_1 = mapGetRatio_Exp4(fph1, fph2)
        #without amplification
        #if gl_exp_2_parameters_p:
        #    frac_1 = mapGetRatio(ph1, ph2)           
        #else:
        #    frac_1 = mapGetRatio_Exp4(ph1, ph2)
        ratio_ar.append(frac_1)
        ants_1 = gl_ants_per_m * frac_1
        ants_2 = gl_ants_per_m * (1.0 - frac_1)
        ph1_next = ph1 * math.exp(gl_decay*1.1)
        ph2_next = ph2 * math.exp(gl_decay*1.1)
        if f1 is True:
            ph1_next += ants_1 * gl_deposit_rate_exploit
        else:
            ph1_next += ants_1 * gl_deposit_rate_explore
        if f2 is True:
            ph2_next += ants_2 * gl_deposit_rate_exploit
        else:
            ph2_next += ants_2 * gl_deposit_rate_explore
        print('{0}   (ph1: {1:1.3f}, ph2: {2:1.3f})    (fph1: {3:1.3f}, fph2: {4:1.3f})  frac_1: {5:1.3f}'.format(m, ph1, ph2, fph1, fph2, frac_1))
              
        ph1 = max(ph1_next, gl_noise_level)
        ph2 = max(ph2_next, gl_noise_level)
    return(ph1, ph2)


# Parameteters better tuned to match the branch preference trajectory over time
# in Dussutor's Experiment 4, "dynamic environment".
# The ratio map is a function of the distance to the midline fbr1 = fbr2
# and also the distance to the origin fbr1 = fbr2 = 0.
# There is a radius of about 8.
# Above this radius, the ratio can go up to 1.
# Below this radius, the ratio is limited to about 0.7
# Also, the channel width of the distance to midline is slightly different.
def mapGetRatio_Exp4(fbr1, fbr2):
    dist_to_mid_2 = distToMid(fbr1, fbr2)
    dist_to_mid_ratio_factor = fracFromDistToMid_Exp4(dist_to_mid_2)
    dist_to_orig = pow(pow(fbr1, 2) + pow((fbr2), 2), .5)
    #parmeters from Experiment 2    
    #dist_to_orig_ratio_factor = .6 + .4 * 1 / (1 + math.exp(-.8 * (dist_to_orig-8)))  #ch1
    #params that better fit Experiment 4
    dist_to_orig_ratio_factor = .7 + .3 * 1 / (1 + math.exp(-.8 * (dist_to_orig-8)))
    total_ratio = (dist_to_mid_ratio_factor * dist_to_orig_ratio_factor) / 2 + .5
    if fbr2 > fbr1:
        total_ratio = 1.0 - total_ratio
    return total_ratio


#with amplification
def fracFromDistToMid_Exp4(dist):
    #parmeters from Experiment 2
    #frac = 1.0 / (1 + math.exp(-4*(dist-2))) - 1 / (1 + math.exp(8))    # ch1
    #params that better fit Experiment 4
    frac = 1.0 / (1 + math.exp(-4*(dist-1))) - 1 / (1 + math.exp(4)) 
    return frac


# These parameters were adjusted from amplifySignal() to better fit Experiment 4.
def amplifySignal_Exp4(x):
    xpr = x - gl_noise_level
    val = 1.4*pow(xpr, .75)
    return val



# Top level function.
# This simulates the condition of Fig. 10
# Food appears at the left branch for gl_trial_duration (45 minutes),
# then moves to the right branch for 45 minutes,
# then moves back to the left branch for 45 minutes.
def simulateDynamic():
    duration = gl_trial_duration
    ratio_ar = []
    ph1 = 0
    ph2 = 0    
    (ph1, ph2) = simulateYJunctionTravel(ph1, ph2, True, False, ratio_ar)
    (ph1, ph2) = simulateYJunctionTravel(ph1, ph2, False, True, ratio_ar)
    (ph1, ph2) = simulateYJunctionTravel(ph1, ph2, True, False, ratio_ar)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(ratio_ar, color="r")    
    plt.xlim([0, duration*3])
    plt.show()


def simulateOnOff():
    duration = gl_trial_duration
    ratio_ar = []
    ph1 = 0
    ph2 = 0
    (ph1, ph2) = simulateYJunctionTravel(ph1, ph2, True, False, ratio_ar)
    (ph1, ph2) = simulateYJunctionTravel(ph1, ph2, False, False, ratio_ar)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(ratio_ar)
    plt.xlim([0, duration*2])
    plt.show()
