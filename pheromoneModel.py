
#pheromoneModel.py is for testing alternative hypotheses to the two-pheromnes
#claim put forth in:
#"The role of multiple pheromones in food recruitment by ants"
#A. Dussutour, S. C. Nicolis, G. Shephard, M. Beekman, and D. J. T. Sumpter
#J Exp Biol. 2009;212: 2337–2348. doi:10.1242/jeb.029827
#
#Our paper based on the model in this module is:
#"A Single-Pheromone Model Accounts for Empirical Patterns of Ant Colony Foraging
# Previously Modeled Using Two Pheromones"
#
#Our one pheromone model proposes that ants in the Dussutour experiment put down the same
#kind of pheromone whether exploring or exploiting / gathering.  But they put down more
#when exploiting, to leave a stronger trail for where to go.  We'll assume that pheromone
#decays exponentially with time.
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
#This file also contains functions for optimizing the free parameters of the model.


############################################################
#
#How to run:
#

#This runs in Python 3

#>>> import pheromoneModel as ph

#Plot fraction of E+F branch as a function of time using the lookup table data
#obtained from Fig. 2.   This is not a parametric model, but it is a preliminary
#test showing that branch preference can be indexed by a single pheromone amount
#parameter.
#>>> ph.runExp12_LookupModel()


#Plot the time course of hypothesized physical pheromone, amplifed pheromone
#measurement, and branch preference, based on the amplification + sigmoid
#product analytical model.
#>>> ph.runExp12_AnalyticalModel(<parameter-bundle>)


#Plot branch preference over time given two swaps in the branch leading
#to food, using the analytical model.  This predicts the preference curves
#produced by Dussutour et al. Experiment 4 "dynamic" condition.
#>>> ph.simulateDynamic(<parameter-bundle>)

#Some parameter bundles are provided:
#gl_initial_parameter_bundle
#gl_parameter_bundle_opt_12
#gl_parameter_bundle_opt_4
#gl_parameter_bundle_opt_124


#Optimize parameters for data from Dussutour Experiments 1&2
#>>> ph.findOptimalParameters_Exp12_AnalyticalModel()

#Optimize parameters for data from Dussutour Experiment 4
#>>> ph.findOptimalParameters_Exp4()

#Optimize parameters for data from Dussutour Experiments 1&2 and Experiment 4 jointly
#>>> ph.findOptimalParameters_Exp12_Exp4()

#The tradeoffs in optimizations to some extent hinge on the parameter bounds set
#in gl_param_lower_bounds and gl_param_upper_bounds.
#More work could be done to explore how these bounds affect the parameter optima found.


#
#
############################################################


import matplotlib.pyplot as plt
import math
import scipy.optimize as optimize
import numpy as np
from scipy.optimize import Bounds



############################################################
#
#Empirical data
#

# The preference table tells the proportion of ants choosing the E+F branch in
# two conditions, E+F vs. N (Experiment 1), and E+F vs. E (Experiment 2).
# These preference values are the curves in Fig. 2.
# Each increment is 5 time steps.
# time axis           5    10   15   20   25   30   35   40   45   50
#                    55    60   65   70   75   80   85   90   95   100
#                   105   110  115  120

# Experiment 1 behavioral results 
# fraction of ants choosing the E+F branch vs. the N branch.
gl_frac_table_n = [.94, .93, .89, .81, .78, .78, .80, .78, .76, .77,
                   .73, .75, .74, .70, .66, .65, .54, .54, .48, .50,
                   .50, .51, .50, .49]


# Experiment 2 behavioral results
# fraction of ants choosing the E+F branch vs. the E branch
gl_frac_table_e = [.94, .91, .83, .75, .68, .62, .56, .51, .52, .49,
                   .50, .50, .50, .49, .51, .50, .50, .50, .51, .50,
                   .48, .50, .49, .50]

# These empirical behavioral preferences are converted to a lookup table
# that tells branch preference ratio as a function of amount of pheromone on each branch. 
# The conversion assumes an exponential decay in the amount of pheromone.




#This data for Dussuotour etal Experiment 4 is read off from their Fig. 10.
#The first entry is for t=0, imputed to be the first plotted entry which is t = 1.
#                       0    1    2    3    4    5    6    7    8    9
#                      10   11   12   13   14   15   16   17   18   19
#                      20   21   22   23   24   25   26   27   28   29

gl_frac_table_exp4 = [.57, .57, .60, .56, .52, .54, .68, .62, .68, .72,   #0-9
                      .69, .70, .70, .75, .76, .82, .81, .87, .84, .83,   #10-19
                      .85, .85, .89, .88, .89, .91, .88, .92, .89, .91,   #20-29
                      .92, .91, .89, .90, .93, .92, .91, .91, .93, .92,   #30-39
                      .93, .90, .92, .91, .94, .94, .89, .88, .84, .80,   #40-49
                      .78, .74, .74, .74, .69, .67, .64, .63, .57, .57,   #50-59
                      .49, .44, .40, .45, .40, .46, .40, .40, .37, .32,   #60-69
                      .30, .31, .29, .29, .27, .28, .29, .24, .24, .25,   #70-79
                      .24, .22, .22, .23, .24, .23, .20, .18, .20, .20,   #80-89
                      .19, .28, .28, .33, .36, .52, .50, .48, .49, .54,   #90-99
                      .57, .57, .58, .65, .67, .71, .72, .70, .73, .75,   #100-109
                      .76, .77, .74, .76, .78, .77, .78, .82, .84, .83,   #110-119
                      .82, .83, .85, .84, .85, .86, .86, .85, .85, .86,   #120-129
                      .85]                                                #130
                      

def plotFracTable(frac_table = None):
    if frac_table == None:
        frac_table = gl_frac_table_exp4

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(frac_table)
    #plt.xlim([0, duration*2])
    plt.show()


#
#
############################################################ empirical data


############################################################
#
#Free parameters are kept in a dict or array format called a parameter_bundle.
#Some parameter_bundles hold parameters estimated manually.
#Then, values in parameter_bundles can be tuned by numerical optimization.
#

#provide min and max noise levels for estimating parameters, especially for the dynamic
#case of Experiment 4.

#Clamp the noise level, removing it as a free parameter.
#This estimate comes from the initial evaluation of Experiemnts 1 & 2, in Fig. 4.
gl_min_noise_level = .02
gl_max_noise_level = .02001  #some versions of scipy don't allow min and max bounds to be equal

gl_min_amp_C = .001

#Bounds on max pheromone amount and min and max deposit rate seem to be necessary to keep
#Exp4 from driving these to extreme values under parameter optimization.

gl_min_c1 = .001
gl_max_c1 = 10.0

gl_min_c2 = .001
gl_max_c2 = 10.0

gl_min_deposit_rate_exploit = .001   # pheromone per ant deposited
gl_min_deposit_rate_explore = .001   # pheromone per ant deposited

gl_max_deposit_rate_exploit = 1.0   # pheromone per ant deposited
gl_max_deposit_rate_explore = 1.0   # pheromone per ant deposited



# c1 is the assumed amount of pheromone on the E+F branch at the start
# c2 is the assumed amount of pheromone on the E branch at the start
# Decay of c1 to the noise level seems to occur after 90 steps.
# Call the noise level 0.02.
# this gives a decay rate of -.069
# 10 * exp(90 * -.069) = 0.02


def figureDecayRate(c1=10, t=90, noise_level = .02):
    #print('c1: ' + str(c1) + ' noise_level: ' + str(noise_level))
    return math.log(noise_level/c1) / t


gl_decay_time = 90.0

#This is no longer used in the simulation functions.
#Instead, decay rate is computed from c1, noise level, and gl_decay_time.
gl_decay = -.069
# Estimated from the data, how long it takes
# E+F vs. N to reach the pheromone noise level,
# starting from c1 = 10.



####################
#
#Older method for setting parameters by global variables
#These parameters are pretty much not used any more, but are kept for reference
#and just in case.
#

gl_noise_level = .02
# minimum signal level, initial estimate of 0.02


gl_amp_A = 5.5
gl_amp_B = gl_noise_level
gl_amp_C = .25

    
#parameters of the distance-to-midline and distance-from-origin sigmoid functions
#of the single-pheromone model, corresponding to terms in the paper.
# These parameters were estimated from plot of the observed data from Fig. 2.
gl_s_m = 4.0
gl_d_0_m = 2.0
#gl_m_z = 8.0  #should be calculated from s_m and d_0_m
gl_s_0 = 0.6
gl_s_r = 0.8
gl_d_0_r = 8.0

gl_deposit_rate_exploit = .011    # pheromone per ant deposited, Exp. 4
gl_deposit_rate_explore = .0011   # pheromone per ant deposited, Exp. 4

#
#
####################



#
#
############################################################ loose parameters


############################################################
#
#Parameter bundles
#


gl_parameter_index_dict = {'c1': 0,
                           'c2': 1,  
                           'amp_A': 2,
                           'noise_level': 3,
                           'amp_C': 4,
                           's_m': 5,
                           'd_0_m': 6,
                           's_0': 7,
                           's_r': 8,
                           'd_0_r': 9,
                           'rate_exploit': 10,
                           'rate_explore': 11}
    

#parameter_bundle initially found by manual adjustment, per pheromones2.py
gl_initial_parameter_bundle = {'c1': 10, 
                               'c2': 1,  
                               'amp_A': 5.5,
                               'noise_level': .02,  #amp_B
                               'amp_C': .25,
                               's_m': 4.0,
                               'd_0_m': 2.0,
                               's_0': 0.6,
                               's_r': 0.8,
                               'd_0_r': 8.0,
                               'rate_exploit': .011,
                               'rate_explore': .0011}                            


#These are parameters found by running findOptimalParameters_Exp12_AnalyticalModel()
#[9.99561681e+00, 3.99999984e-02, 6.44159619e+00, 2.00000000e-02, 1.42554956e-01,
# 5.33265636e+00, 2.92366609e+00, 5.11085502e-01, 1.53755136e+00, 7.83079870e+00,
# 1.10000000e-02, 1.10000000e-03]
#fun: 0.011505089101438139
#2022/12/02
gl_parameter_bundle_opt_12 =  {'c1': 10.0,
                               'c2': .04,
                               'amp_A': 6.44,
                               'noise_level': .02,  #amp_B
                               'amp_C': .143,
                               's_m': 5.33,
                               'd_0_m': 2.92,
                               's_0': 0.511,
                               's_r': 1.54,
                               'd_0_r': 7.83,
                               'rate_exploit': .011,
                               'rate_explore': .0011}


#These are parameters found by running
#findOptimalParameters_Exp4(gl_initial_parameter_bundle)
#with noise_level clamped to 0.02.
#[1.00000000e+01, 1.00000000e+00, 2.27435659e+00, 2.00000000e-02, 2.63689725e-01,
# 6.87175514e-01, 2.09627981e+00, 2.60883899e-01, 1.93488570e+00, 9.94319314e+00,
# 1.00000000e+00, 1.00000000e-03]
#exp4 score function:  fun: 0.8176292217031855
gl_parameter_bundle_opt_4 =  {'c1': 10.0,          #not applicable to Experiment 4
                              'c2': 1.0,           #not applicable to Experiment 4
                              'amp_A': 2.27,
                              'noise_level': .02,  #amp_B, clamped
                              'amp_C': .264,
                              's_m': .687,
                              'd_0_m': 2.1,
                              's_0': .261,
                              's_r': 1.93,
                              'd_0_r': 9.4,
                              'rate_exploit': 1.0,
                              'rate_explore': .001}


#These are parameters found by running findOptimalParameters_Exp12_Exp4()
#which jointly optimizes parameters for Dussutour etal Experiments 1&2 and Experiment 4.
#This optimization includes a weighting of .35 on the exp4 term to even out the number
#of time points considered. These parameters give slightly worse fits to each experiment,
#but is a good compromise between them.
#noise_level is clamped at 0.02
#Bounds were placed on c1, c2, exploit and explore and rates
#[1.00000000e+01, 1.55469755e+00, 4.92641508e+00, 2.00000000e-02, 2.97714774e-01,
# 1.58243603e+00, 1.61971247e+00, 6.17344640e-01, 1.81683308e+00, 7.71275223e+00,
# 8.16419469e-03, 1.00000000e-03]
#fun: 0.5214666920149161
gl_parameter_bundle_opt_124 =  {'c1': 10.0,          #applicable only to Experiment 1/2
                                'c2': 1.55,          #applicable only to Experiment 1/2
                                'amp_A': 4.92,
                                'noise_level': .02,  #amp_B, clamped
                                'amp_C': .298,
                                's_m': 1.58,
                                'd_0_m': 1.61,
                                's_0': .617,
                                's_r': 1.81,
                                'd_0_r': 7.71,
                                'rate_exploit': .0082,
                                'rate_explore': .001}



#bounds in parameter_bundle format
gl_param_lower_bounds = [gl_min_c1, gl_min_c2,
                         .001, gl_min_noise_level, gl_min_amp_C,
                         .001, .001, .001, .001, .001, 
                         gl_min_deposit_rate_exploit,
                         gl_min_deposit_rate_explore]


gl_param_upper_bounds = [gl_max_c1, gl_max_c2,
                         np.inf, gl_max_noise_level, 40.0,
                         np.inf, np.inf, np.inf, np.inf, np.inf,
                         gl_max_deposit_rate_exploit,
                         gl_max_deposit_rate_explore]



gl_parameter_bounds = Bounds(gl_param_lower_bounds, gl_param_upper_bounds)


def parameterBundleToArray(parameter_bundle):
    if type(parameter_bundle) is list:
        return parameter_bundle
    elif type(parameter_bundle) is np.ndarray:
        return parameter_bundle
    
    c1 = parameter_bundle.get('c1')
    c2 = parameter_bundle.get('c2')
    amp_A = parameter_bundle.get('amp_A')    
    noise_level = parameter_bundle.get('noise_level')
    amp_C = parameter_bundle.get('amp_C')
    s_m = parameter_bundle.get('s_m')
    d_0_m = parameter_bundle.get('d_0_m')
    s_0 = parameter_bundle.get('s_0')
    s_r = parameter_bundle.get('s_r')
    d_0_r = parameter_bundle.get('d_0_r')
    rate_exploit = parameter_bundle.get('rate_exploit')
    rate_explore = parameter_bundle.get('rate_explore')    
    pb_array = [c1, c2, amp_A, noise_level, amp_C, s_m, d_0_m, s_0, s_r, d_0_r,
                rate_exploit, rate_explore]    
    return pb_array


def parameterBundleArrayToDict(parameter_bundle):
    if type(parameter_bundle) is dict:
        return parameter_bundle
    parameter_bundle_seq = parameter_bundle
    pb_dict = {}
    pb_dict['c1'] = parameter_bundle_seq[gl_parameter_index_dict.get('c1')]
    pb_dict['c2'] = parameter_bundle_seq[gl_parameter_index_dict.get('c2')]    
    pb_dict['amp_A'] = parameter_bundle_seq[gl_parameter_index_dict.get('amp_A')]
    pb_dict['noise_level'] = parameter_bundle_seq[gl_parameter_index_dict.get('noise_level')]
    pb_dict['amp_C'] = parameter_bundle_seq[gl_parameter_index_dict.get('amp_C')]
    pb_dict['s_m'] = parameter_bundle_seq[gl_parameter_index_dict.get('s_m')]
    pb_dict['d_0_m'] = parameter_bundle_seq[gl_parameter_index_dict.get('d_0_m')]
    pb_dict['s_0'] = parameter_bundle_seq[gl_parameter_index_dict.get('s_0')]
    pb_dict['s_r'] = parameter_bundle_seq[gl_parameter_index_dict.get('s_r')]
    pb_dict['d_0_r'] = parameter_bundle_seq[gl_parameter_index_dict.get('d_0_r')]
    pb_dict['rate_exploit'] = parameter_bundle_seq[gl_parameter_index_dict.get('rate_exploit')]
    pb_dict['rate_explore'] = parameter_bundle_seq[gl_parameter_index_dict.get('rate_explore')] 
    return pb_dict


#parameter_bundle can be either the array or the dict form
def printParameterBundle(parameter_bundle):
    if type(parameter_bundle) is list:
        parameter_bundle_dict = parameterBundleArrayToDict(parameter_bundle)
    elif type(parameter_bundle) is dict:
        parameter_bundle_dict = parameter_bundle
    print('c1: ' + str(parameter_bundle_dict.get('c1')) + '  c2: ' + str(parameter_bundle_dict.get('c2')))
    print('amp_A: ' + str(parameter_bundle_dict.get('amp_A')) + '  noise_level: ' + str(parameter_bundle_dict.get('noise_level')) + '  amp_C: ' + str(parameter_bundle_dict.get('amp_C')))
    print('s_m: ' + str(parameter_bundle_dict.get('s_m')) + '  d_0_m: ' + str(parameter_bundle_dict.get('d_0_m')) + '  s_0: ' + str(parameter_bundle_dict.get('s_0')) +'  s_r: ' + str(parameter_bundle_dict.get('s_r')) +'  d_0_r: ' + str(parameter_bundle_dict.get('s_m')) +'  s_m: ' + str(parameter_bundle_dict.get('d_0_r')) + ' rate_exploit: ' + str(parameter_bundle_dict.get('rate_exploit')) + ' rate_explore: ' + str(parameter_bundle_dict.get('rate_explore')))


#
#
############################################################ parameter bundles


############################################################
#
#This section is the initial demonstration that a single parameter (pheromone)
#could index a tabular function to generate observed data.
#This is a precursor to the final analytical model.
#

#Set up a table for looking up fraction of ants preferring Branch A as a function of
#ph1 and ph2, assuming some starting amounts of ph1 and ph2, and exponential decay over time.
#The table is represented just as a list of entries of the form,  (ph1, ph2, frac),
#with values of frac coming from the two curves of Dussutour Fig. 2.
def setupTable(c1=10, c2=1):
    lut_ph1_ph2_frac = []  # (ph1, ph2, frac)
    c1 = c1 - gl_noise_level
    c2 = c2 - gl_noise_level
    global gl_decay
    gl_decay = figureDecayRate(c1, gl_decay_time, gl_noise_level)
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


def setTable(c1=10):
    global gl_lut
    gl_lut = setupTable(c1, 1)


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
    br1_ar = []   # exponential decaying pheromone on Branch A
    br2a_ar = []  # exponential decaying pheromone on Branch B
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


# Plot Lookup table of preference fraction vs. measured pheromone on 2 branches
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


#
#
############################################################ lookup table



############################################################
#
#Analytical model
#
    
    

#power law amplification of raw pheromone x
#amp_A is scaling term
#amp_B is noise level offset from raw pheromone
#amp_C is power
def amplifySignal(x, amp_A = gl_amp_A, amp_B = gl_amp_B, amp_C = gl_amp_C):
    amp_signal = amp_A*pow((x - amp_B), amp_C)
    #print('x: ' + str(x) + ' amp_A: ' + str(amp_A) + ' amp_B: ' + str(amp_B) + ' amp_C: ' + str(amp_C) + ' amp_signal: ' + str(amp_signal))
    if math.isnan(amp_signal):
        print('problem')
        print('x: ' + str(x) + ' amp_A: ' + str(amp_A) + ' amp_B: ' + str(amp_B) + ' amp_C: ' + str(amp_C) + ' amp_signal: ' + str(amp_signal))
        aa = 23/0
    return amp_signal

    
# These parameters were estimated by plotting the observed data from Fig. 2
# and adjusting the parameters until the preference ratio transition occurred
# at about the same diagonal for E+F vs. E condition and E+F vs. N condition.
#def amplifySignal(x):
#    xpr = x - gl_noise_level
#    val = 5.5*pow(xpr, .25)
#    return val



# Returns distance to the midline of equal pheromone amounts.
# As everywhere, units are arbitrary pheromone amounts, assuming the
# E+F branch starts with pheromone amount 10.
def distToMid(xpr, ypr):
    xm = (xpr + ypr)/2
    ym = xm
    dist = pow(pow(xpr-xm, 2) + pow((ypr-ym), 2), .5)
    return dist


# This squashing function creates a band of equal preference near the midline,
# transitioning to a preference for one branch or the other at a distance of
# about 2 from the midline.
def fracFromDistToMid(dist, s_m = gl_s_m, d_0_m = gl_d_0_m):
    exp_d0 = s_m * d_0_m
    d0_offset = 1.0 / (1 + math.exp(exp_d0))  #value at d=0 to subtract so the function returns .5 for d=0
    exponent = -s_m*(dist-d_0_m)
    if abs(exponent) > 600:
        exponent = np.sign(exponent) * 600
    frac = 1.0 / (1 + math.exp(exponent)) - d0_offset
    return frac




#fbr1 and fbr2 are the amplified pheromone amounts on Branch A and Branch B respectively.
# The ratio map is a function of the distance to the midline fbr1 = fbr2
# and also the distance to the origin fbr1 = fbr2 = 0.
# There is a radius of about 8.
# Above this radius, the ratio can go up to 1.0
# Below this radius, the ratio is limited to about 0.6.
def mapGetRatio(fbr1, fbr2, s_m = gl_s_m, d_0_m = gl_d_0_m, s_0 = gl_s_0, s_r = gl_s_r, d_0_r = gl_d_0_r):
                
    dist_to_mid_2 = distToMid(fbr1, fbr2)
    dist_to_mid_ratio_factor = fracFromDistToMid(dist_to_mid_2, s_m, d_0_m)
    dist_to_orig = pow(pow(fbr1, 2) + pow((fbr2), 2), .5)
    exponent = -s_r * (dist_to_orig-d_0_r)
    if abs(exponent) > 600:
        exponent = np.sign(exponent) * 600
    dist_to_orig_ratio_factor = s_0 + (1.0 - s_0) / (1 + math.exp(exponent))
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
# This assumes a single pheromone starting with concentrations c1, c2, and noise_level,
# for the E+F, E, and N branches, respectively.
def runExp12_AnalyticalModel(parameter_bundle = gl_initial_parameter_bundle):
    parameter_bundle = parameterBundleArrayToDict(parameter_bundle)
    
    c1 = parameter_bundle.get('c1')
    c2 = parameter_bundle.get('c2')
    amp_A = parameter_bundle.get('amp_A')
    noise_level = parameter_bundle.get('noise_level')
    noise_level = max(gl_min_noise_level, noise_level)
    amp_C = parameter_bundle.get('amp_C')
    amp_C = max(gl_min_amp_C, amp_C)
    s_m = parameter_bundle.get('s_m')
    d_0_m = parameter_bundle.get('d_0_m')
    s_0 = parameter_bundle.get('s_0')
    s_r = parameter_bundle.get('s_r')
    d_0_r = parameter_bundle.get('d_0_r')

    c1 = c1 - noise_level
    c2 = c2 - noise_level
    decay_time = 90

    decay_rate = figureDecayRate(c1, decay_time, noise_level)
    duration = 120
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    br1_ar = []   # exponential decaying pheromone on branch A
    br2a_ar = []  # exponential decaying pheromone on branch B
    fbr1_ar = []
    fbr2a_ar = []
    fbr2b_ar = []
    ratio2a_ar = []
    ratio2b_ar = []
    for m in range(duration):   # duration in minutes
        br1 = c1 * math.exp(decay_rate * m) + noise_level
        br2a = c2 * math.exp(decay_rate * m) + noise_level
        br2b = noise_level
        #print('m: ' + str(m) + ' br1: ' + str(br1) + ' br2a: ' + str(br2a) + ' br2b: ' + str(br2b))
        fbr1 = amplifySignal(br1, amp_A, noise_level, amp_C)
        fbr2a = amplifySignal(br2a, amp_A, noise_level, amp_C)
        fbr2b = amplifySignal(br2b, amp_A, noise_level, amp_C)        
        br1_ar.append(br1)
        br2a_ar.append(br2a)
        fbr1_ar.append(fbr1)
        fbr2a_ar.append(fbr2a)
        fbr2b_ar.append(fbr2b)
        ratio_2a = mapGetRatio(fbr1, fbr2a, s_m, d_0_m, s_0, s_r, d_0_r)
        ratio_2b = mapGetRatio(fbr1, fbr2b, s_m, d_0_m, s_0, s_r, d_0_r)                
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


#Compute a sum squared error score for predictions from the analytical model under the
#parameters passed, compared with empirical data from Dussutour Fig. 2.
#parameter_bundle is a list of parameters indexed by gl_parameter_index_dict
def exp12ScoreFunction(parameter_bundle):    
    if type(parameter_bundle) is dict:
        parameter_bundle_ar = parameterBundleToArray(parameter_bundle)
    elif type(parameter_bundle) is list:
        parameter_bundle_ar = parameter_bundle
    elif type(parameter_bundle) is np.ndarray:
        parameter_bundle_ar = parameter_bundle
    else:
        print('unknown type ' + str(type(parameter_bundle)) + ' for parameter_bundle: ' + str(parameter_bundle))
        
    print('pb: ' + str(parameter_bundle_ar))
    c1 = parameter_bundle_ar[gl_parameter_index_dict.get('c1')]
    c2 = parameter_bundle_ar[gl_parameter_index_dict.get('c2')]
    amp_A = parameter_bundle_ar[gl_parameter_index_dict.get('amp_A')]
    noise_level = parameter_bundle_ar[gl_parameter_index_dict.get('noise_level')]
    noise_level = max(gl_min_noise_level, noise_level)
    amp_C = parameter_bundle_ar[gl_parameter_index_dict.get('amp_C')]
    amp_C = max(gl_min_amp_C, amp_C)
    s_m = parameter_bundle_ar[gl_parameter_index_dict.get('s_m')]
    d_0_m = parameter_bundle_ar[gl_parameter_index_dict.get('d_0_m')]
    s_0 = parameter_bundle_ar[gl_parameter_index_dict.get('s_0')]
    s_r = parameter_bundle_ar[gl_parameter_index_dict.get('s_r')]
    d_0_r = parameter_bundle_ar[gl_parameter_index_dict.get('d_0_r')]

    c1 = c1 - noise_level
    c2 = c2 - noise_level
    c1 = max(c1, noise_level)
    c2 = max(c2, noise_level)
    decay_time = 90
    decay_rate = figureDecayRate(c1, decay_time, noise_level)
    sum_sq_error = 0
    for m1 in range(1, 25):
        m5 = m1*5
        br1 = c1 * math.exp(decay_rate * m5) + noise_level  #exponential decaying pheromone on branch A
        br2a = c2 * math.exp(decay_rate * m5) + noise_level #exponential decaying pheromone on branch B
        br2b = noise_level
        fbr1 = amplifySignal(br1, amp_A, noise_level, amp_C)
        fbr2a = amplifySignal(br2a, amp_A, noise_level, amp_C)
        fbr2b = amplifySignal(br2b, amp_A, noise_level, amp_C)
        ratio_2a = mapGetRatio(fbr1, fbr2a, s_m, d_0_m, s_0, s_r, d_0_r)
        ratio_2b = mapGetRatio(fbr1, fbr2b, s_m, d_0_m, s_0, s_r, d_0_r)

        e1 = ratio_2a - gl_frac_table_e[m1-1]
        e2 = ratio_2b - gl_frac_table_n[m1-1]
        sum_sq_error += e1*e1 + e2*e2
        #print('m5: ' + str(m5) + ' ratio_2a: ' + f'{ratio_2a:.3f}' + '  table_e: ' + str(gl_frac_table_e[m1]) + '   ratio_2b: ' + f'{ratio_2b:.3f}' + '  table_n: ' + str(gl_frac_table_n[m1]))
    return sum_sq_error


#run scipy.optimize.minimize to find parameters that best fit preditions of the 1PM analytical model
#to branch preference curves reported by Dussutour et al. in their Fig. 2.
def findOptimalParameters_Exp12_AnalyticalModel(initial_parameter_bundle = gl_initial_parameter_bundle):
    init_pb_ar = parameterBundleToArray(initial_parameter_bundle)

    res_pb = optimize.minimize(exp12ScoreFunction, init_pb_ar, method=gl_optimization_method,
                               bounds=gl_parameter_bounds)    

    return res_pb
    
    
#
#
##################################

    
    
##################################
#
# Simulating Experiment 4, "dynamic environment" alternating food placement.
#

# We can use with the parameters used for Experiment 1-2, but the plot has
# some differences in appearance from Fig. 10.
# We can adjust the parameters to better fit Fig. 10, but then there is not
# as close a fit to Fig. 2.  Really, we would need more experimental data to
# find the right functional form and parameters to fit both figures well and
# predict other experimental results about ant behavior in this kind of setup.

gl_ants_per_m = 50     #based on Dussutour paper

gl_trial_duration = 45   #Exp 4, minutes per branch getting food 


#Simulate ant behavior for gl_ants_per_m ants per minute exploring a Y-maze.
#The maze starts off with c1 and c2 levels of pheromone, respectively, on Branch A and Branch B.
#f1 and f2 are flags for whether each branch has food.
#If a branch has food, then ants will deposit pheromone at a rate, deposit_rate_exploit.
#If a branch does not have food, then ants will deposit pheromone at a rate, deposit_rate_explore.
#
#This appends ph1/ph2 ratios to ratio_ar.
#Returns two values, ph1 and ph2, which are pheromone concentrations on
#Branches A and B respectively after gl_trial_duration minutes.
def simulateYJunctionTravel(c1=0, c2=0, f1=False, f2=False, ratio_ar=None,
                            parameter_bundle = gl_parameter_bundle_opt_4):
    if type(parameter_bundle) is dict:
        parameter_bundle_ar = parameterBundleToArray(parameter_bundle)
    elif type(parameter_bundle) is list:
        parameter_bundle_ar = parameter_bundle
    elif type(parameter_bundle) is np.ndarray:
        parameter_bundle_ar = parameter_bundle
    else:
        print('unknown type ' + str(type(parameter_bundle)) + ' for parameter_bundle: ' + str(parameter_bundle))
        
    duration = gl_trial_duration
    #c1_baseline is used to figure the decay rate for the exp4 simulation
    c1_baseline = parameter_bundle_ar[gl_parameter_index_dict.get('c1')]
    amp_A = parameter_bundle_ar[gl_parameter_index_dict.get('amp_A')]
    noise_level = parameter_bundle_ar[gl_parameter_index_dict.get('noise_level')]
    noise_level = max(gl_min_noise_level, noise_level)
    amp_C = parameter_bundle_ar[gl_parameter_index_dict.get('amp_C')]
    amp_C = max(gl_min_amp_C, amp_C)
    s_m = parameter_bundle_ar[gl_parameter_index_dict.get('s_m')]
    d_0_m = parameter_bundle_ar[gl_parameter_index_dict.get('d_0_m')]
    s_0 = parameter_bundle_ar[gl_parameter_index_dict.get('s_0')]
    s_r = parameter_bundle_ar[gl_parameter_index_dict.get('s_r')]
    d_0_r = parameter_bundle_ar[gl_parameter_index_dict.get('d_0_r')]
    deposit_rate_exploit = parameter_bundle_ar[gl_parameter_index_dict.get('rate_exploit')]
    deposit_rate_exploit = max(deposit_rate_exploit, gl_min_deposit_rate_exploit)
    deposit_rate_explore = parameter_bundle_ar[gl_parameter_index_dict.get('rate_explore')]
    deposit_rate_explore = max(deposit_rate_explore, gl_min_deposit_rate_explore)
    
    c1 = c1 - noise_level
    c2 = c2 - noise_level
    decay_time = 90
    c1_baseline = max(c1_baseline, noise_level)
    decay_rate = figureDecayRate(c1_baseline, decay_time, noise_level)
    print('decay_rate: ' + str(decay_rate))

    ph1 = max(c1, noise_level)
    ph2 = max(c2, noise_level)

    if ratio_ar is None:
        ratio_ar = []
    for m in range(duration):   # duration in minutes
        fph1 = amplifySignal(ph1, amp_A, noise_level, amp_C)
        fph2 = amplifySignal(ph2, amp_A, noise_level, amp_C)
        frac_1 = mapGetRatio(fph1, fph2, s_m, d_0_m, s_0, s_r, d_0_r)
        ratio_ar.append(frac_1)
        ants_1 = gl_ants_per_m * frac_1
        ants_2 = gl_ants_per_m * (1.0 - frac_1)
        ph1_next = ph1 * math.exp(decay_rate) 
        ph2_next = ph2 * math.exp(decay_rate)
        if f1 is True:
            ph1_next += ants_1 * deposit_rate_exploit
        else:
            ph1_next += ants_1 * deposit_rate_explore
        if f2 is True:
            ph2_next += ants_2 * deposit_rate_exploit
        else:
            ph2_next += ants_2 * deposit_rate_explore
        #print('{0}   (ph1: {1:1.3f}, ph2: {2:1.3f})    (fph1: {3:1.3f}, fph2: {4:1.3f})  frac_1: {5:1.3f}'.format(m, ph1, ph2, fph1, fph2, frac_1))
              
        ph1 = max(ph1_next, noise_level)
        ph2 = max(ph2_next, noise_level)
    return(ph1, ph2)



# Top level function.
# This simulates the condition of Fig. 10
# Food appears at the left branch for gl_trial_duration (45 minutes),
# then moves to the right branch for 45 minutes,
# then moves back to the left branch for 45 minutes.
def simulateDynamic(parameter_bundle = gl_parameter_bundle_opt_4, plot_p=True):
    duration = gl_trial_duration
    ratio_ar = []
    ph1 = 0
    ph2 = 0 
    (ph1, ph2) = simulateYJunctionTravel(ph1, ph2, True, False, ratio_ar, parameter_bundle)
    (ph1, ph2) = simulateYJunctionTravel(ph1, ph2, False, True, ratio_ar, parameter_bundle)
    (ph1, ph2) = simulateYJunctionTravel(ph1, ph2, True, False, ratio_ar, parameter_bundle)
    if plot_p:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(ratio_ar, color="r")    
        plt.xlim([0, duration*3])
        plt.show()
    return ratio_ar



#Simulate just having pheromone present on branch A for 45 minutes, then absent.
def simulateOnOff(parameter_bundle = gl_parameter_bundle_opt_4):
    duration = gl_trial_duration
    ratio_ar = []
    ph1 = 0
    ph2 = 0
    (ph1, ph2) = simulateYJunctionTravel(ph1, ph2, True, False, ratio_ar, parameter_bundle)
    (ph1, ph2) = simulateYJunctionTravel(ph1, ph2, False, False, ratio_ar, parameter_bundle)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(ratio_ar)
    plt.xlim([0, duration*2])
    plt.show()




#Simulate just having pheromone present on both Branch A and Branch B after
#having pheromone on just Branch A.
#The 1PM model converges to 50/50.
#By comparison, Dussutour et al. predict oscillations under some circumstances.
def simulateBothOn(parameter_bundle = gl_parameter_bundle_opt_4, n_both_on_segments=4):
    duration = gl_trial_duration
    ratio_ar = []
    ph1 = 0
    ph2 = 0
    (ph1, ph2) = simulateYJunctionTravel(ph1, ph2, True, False, ratio_ar, parameter_bundle)
    for i in range(n_both_on_segments):
        (ph1, ph2) = simulateYJunctionTravel(ph1, ph2, True, True, ratio_ar, parameter_bundle)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(ratio_ar)
    plt.xlim([0, duration*(1+n_both_on_segments)])
    plt.ylim([0, 1])
    plt.show()


gl_optimization_method = 'L-BFGS-B'


#run scipy.optimize.minimize to find parameters that best fit predicted branch preference curves
#over time for Dussutour Experiment 4 data.
#Returns a data structure res_pb containing optimized parameters and other information.
#The parameter array is res_pb.x
#
#The exp4ScoreFunction produces a long manifold that trades off the parameters with one another but
#all with similar low values of the sum_sq_error cost. Different values give somewhat different shapes
#for the preference curve.
#Under some of the parameterizations I used during development, I found convergence to different
#parameter values depending on the starting parameters.  I have not observed this under the current
#parameterization but the possibility of multiple minima cannot be entirely excluded.
#To find the best minimum, it would be best to pass a jacobian. But we are not doing so, so it has to
#be estimated from the function.
def findOptimalParameters_Exp4(initial_parameter_bundle = gl_initial_parameter_bundle, eps=1.0e-08):
    init_pb_ar = parameterBundleToArray(initial_parameter_bundle)

    #I do not believe it necessary to override the defeault parameters for L-BFGS-B minimization.
    #These overrides are left over from development. This works, so I'm just keeping them.
    #Previously, I found that it could be helpful to set epsilon smaller in order to achieve convergence, per
    #https://stackoverflow.com/questions/34663539/scipy-optimize-fmin-l-bfgs-b-returns-abnormal-termination-in-lnsrch
    res_pb = optimize.minimize(exp4ScoreFunction, init_pb_ar, method=gl_optimization_method,
                               tol = 1.0e-12,
                               bounds=gl_parameter_bounds, options={'eps':eps,
                                                                    'ftol':1.0e-12,
                                                                    'gtol':1.0e-8,
                                                                    'maxls': 1000,
                                                                    'maxcor':1000,
                                                                    'maxiter':100000,
                                                                    'maxfun':100000})
    return res_pb


#Compute a sum squared error score for predictions from the analytical model under the
#parameters passed, compared with empirical data from Dussutour Fig. 10.
#parameter_bundle is a list of parameters indexed by gl_parameter_index_dict
def exp4ScoreFunction(parameter_bundle):
    print('exp4ScoreFunction parameters: ' + str(parameter_bundle))
    ratio_ar = simulateDynamic(parameter_bundle, False)
    sum_sq_error = 0
    frac_table = gl_frac_table_exp4  #array of observed preference for branch A vs. branch B
    #for BrachA/BranchB placement of food for 45 minute phases
    for m in range(len(frac_table)):
        error = ratio_ar[m] - frac_table[m]
        sum_sq_error += error*error
    #print('sse: ' + str(sum_sq_error))
    return sum_sq_error


#run scipy.optimize.minimize to find parameters that best fit predicted branch preference curves
#over time for Dussutour Experiments 1&2 and Experiment 4 data jointly.
def findOptimalParameters_Exp12_Exp4(initial_parameter_bundle = gl_initial_parameter_bundle, eps=1.0e-08):
    init_pb_ar = parameterBundleToArray(initial_parameter_bundle)

    res_pb = optimize.minimize(exp12_exp4ScoreFunction, init_pb_ar, method=gl_optimization_method,
                               tol = 1.0e-12,
                               bounds=gl_parameter_bounds, options={'eps':eps,
                                                                    'ftol':1.0e-12,
                                                                    'gtol':1.0e-8,
                                                                    'maxls': 1000,
                                                                    'maxcor':1000,
                                                                    'maxiter':100000,
                                                                    'maxfun':100000})
    return res_pb


#Compute a sum squared error score for predictions from the analytical model under the
#parameters passed, compared with empirical data jointly from both Dussutour Fig. 2 and
#Dussutour Fig. 10.
#A balance factor of .35 is used to equalize the number of samples provided from each experiment.
#parameter_bundle is a list of parameters indexed by gl_parameter_index_dict
#def exp12_exp4ScoreFunction(parameter_bundle, arg0):
def exp12_exp4ScoreFunction(parameter_bundle):
    sse_exp12 = exp12ScoreFunction(parameter_bundle)
    sse_exp4 = exp4ScoreFunction(parameter_bundle)

    total_sse = sse_exp12 + .35 * sse_exp4   #to even out the number of samples considered

    print(' sse_exp12: ' + str(sse_exp12) + ' sse_exp4: ' + str(sse_exp4) + ' total: ' + str(total_sse))
    
    return total_sse


def averageParameterBundles(pb1, pb2):
    if type(pb1) is dict:
        pb1_ar = parameterBundleToArray(pb1)
    else:
        pb1_ar = pb1
    if type(pb2) is dict:
        pb2_ar = parameterBundleToArray(pb2)
    else:
        pb2_ar = pb2
    pb_ave_ar = []
    for i in range(len(pb1_ar)):
        pb_ave_ar.append((pb1_ar[i] + pb2_ar[i])/2.0)
    pb_ave = parameterBundleArrayToDict(pb_ave_ar)
    return pb_ave

  
#
#
############################################################
#
#
################################################################################



