



import matplotlib.pyplot as plt
import math

#import imp as imp
#import pheromones1 as p1
#imp.reload(p1)

#pheromones1.py is for testing alternative hypotheses to the two-pheromnes
#claim by
#"The role of multiple pheromones in food recruitment by ants"
#A. Dussutour, S. C. Nicolis, G. Shephard, M. Beekman, and D. J. T. Sumpter
#downloaded from ResearchGate to /external-docs/Multiplepheromones2009JEB.pdf
#I am skeptical that this experiment proves two pheromones.


#Assume ants put down the same kind of pheromone whether exploring or (exploiting / gathering)
# (which they call foraging, F, but that is a mis-use of the word. "Forage" means search
#  for food as well as gather.)
#But they put down more when exploiting, to leave a stronger trail for where to go.
#We'll assume that pheromone decays exponentially with time.
#
#There are two experimental conditions, the E+F vs E condition, and the E+F vs N condition.
#The E+F branch has pheromone after the ants have both explored and exploited.
#The E branch has pheromone after the ants have only explored only, but not exploited.
#The N branch has no pheromone.
#
#My assessment is that the E+F branch leaves a lot of pheromone, say, 10, and
#the E branch leaves a small but significant amount of pheromone, say, 1.
#So exploiting leaves 10 times as much pheromone as just exploring.
#
#1. When there are good amounts of pheromone, then the measurement of
#which path has more pheromone is clear.  Preferably choose the path with more.
#But the choice preference is not just by ratio of amount of pheromone, it is
#a nonlinear function. If it is clear which branch has more, prefer it no matter
#the ratio. Let's say the function is a sigmoid.
#
#At the start of the experiment, this will take them 95% down the exploit path
#under both conditions, E+F vs. N (no signal) and E+F vs E (weak signal).
#
#2. When there is skimpy pheromone, the pheromone sensor has to amplify
#a very weak signal. After doing so, however, there is still a significant
#difference between pheromone amounts in the E+F vs E branch for a while.
#But after a while, the difference in the amplified signal becomes less
#discernable after about 40 minutes.
#In the E+F vs. N branch, the exponentially decaying pheromone signal still
#persists in the E+F branch. Compared to the noise level in the N branch,
#this difference remains somewhat detectable for about 80 minutes.


#recovering how to hack python after 2 years...
def test1():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    x_ar = []
    y_ar = []
                                       
    for i in range(10):
        print(i)
        x = i
        x_ar.append(i)
        y = pow(x,2.0)
        y_ar.append(y)

    ax1.plot(x_ar)
    ax1.plot(y_ar)
    plt.xlim([0, 10])
    plt.show()


#############################################3
#10/16/2021
#Assume a single pheromone that starts out at concentrations
#c1 and c2 on two branches respectively.
#Their amounts decay exponentially, or in one experiment,
#as the sum of two different exponentials.
#This is exploration of functional forms for the sensory response
#and then preference proportion.
#
#This section includes a lot of flopping around exploring functional forms.
#It was useful to use the Desmos online graphing calculator at 
#https://www.desmos.com/calculator
#


def testExp1(c1, c2, decay = .4):
    duration = 120
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    br1_ar = []   #exponential decaying pheromone on branch 1
    br2a_ar = []  #exponential decaying pheromone on branch 2
    fbr1_ar = []  
    fbr2a_ar = []
    fbr2b_ar = []
    ratio2a_ar = []
    ratio2b_ar = []
                                       
    for m in range(duration):   #minutes
        br1 = c1 * (math.exp(-decay * m))
        br2a = c2 * (math.exp(-decay * m))
        br2b = 0
        #br1 = c1 * (math.exp(-decay * m) + .5 * math.exp(-decay * .5 * m))
        #br2a = c2 * (math.exp(-decay * m) + .5 * math.exp(-decay * .5 * m))
        #br2b = 0
        fbr1 = funct(br1)     #"E+F"
        fbr2a = funct(br2a)    #"E"
        fbr2b = funct(br2b)      #"N"
        diff12a = math.log(abs(fbr1 - fbr2a) + 1)
        diff12a = pow(diff12a, 1.5)
        diff12b = math.log(abs(fbr1 - fbr2b) + 1)
        diff12b = pow(diff12b, 1.5)
        #ratio2a = fbr1 / (fbr1 + fbr2a)
        ratio2a = (.2*diff12a + fbr1) / (fbr1 + fbr2a + diff12a)
        ratiom2a = ratioMap(ratio2a)
        #ratio2b = fbr1 / (fbr1 + fbr2b)
        ratio2b = (.2*diff12b + fbr1) / (fbr1 + fbr2b + diff12b)
        ratiom2b = ratioMap(ratio2b)
        if m < 10:
            print('fbr1: ' + str(fbr1) + ' fbr2a: ' + str(fbr2a) + ' ratio: ' + str(fbr1/fbr2a))
        br1_ar.append(br1)
        br2a_ar.append(br2a)
        fbr1_ar.append(fbr1)
        fbr2a_ar.append(fbr2a)
        fbr2b_ar.append(fbr2b)
        ratio2a_ar.append(ratiom2a)
        ratio2b_ar.append(ratiom2b)
        
    ax1.plot(br1_ar)
    ax1.plot(br2a_ar)
    ax1.plot(fbr1_ar, color="g")
    ax1.plot(fbr2a_ar, color="r")
    ax1.plot(fbr2b_ar, color="b")

    ax2.plot(ratio2a_ar, color="r")
    ax2.plot(ratio2b_ar, color="b")
    plt.xlim([0, duration])
    plt.show()


def ratioMap(x):
    ret = 1.0 / (1.0 + math.exp(-10.0 * (x - .5)))
    #print('ratioMap x: ' + str(x) + ' ret: ' + str(ret))
    return max(0.0, min(1.0, ret))


def funct(x):
    #return math.log(x + 1)
    t1 = .5 * pow(x, .25) + .5 * x + .2
    #t1 = .5 * x + .2
    #t2 = 2 / (1 + math.exp(-2 * t1)) - 1
    #print('x: ' + str(x) + ' t1: ' + str(t1) + ' t2: ' + str(t2))
    #t2 = t1 + 1 / (1 + math.exp(-20*x)) - 1 / (1 + math.exp(-5*x))
    return t1


def funct2(x):
    return x + .01

def funct9(x):
    ret = 1 / (1 + math.exp(-20*x)) - \
          1 / (1 + math.exp(-5*x)) + x + .1
    return ret

def funct10(x):
    ret = 1 / (1 + math.exp(-40*x)) + 1*math.log(x+1.6) -1 + .25
    return ret

def funct1(x):
    ret = 1 / (1 + math.exp(-20*(x+.1))) + .1*x +.5
    return ret


def funct3(x):
    return math.log(x+1) + \
        .05 / (1 + math.exp(-50*x)) - \
        .05 / (1 + math.exp(-2*x)) + .05


def funct4(x):
    #ret = min(1, max(0.01, 1 / (1 + math.exp(-100*x)) - 1 / (1 + math.exp(-2*x)) + .5))
    ret = 1 / (1 + math.exp(-100*x)) - \
          1 / (1 + math.exp(-2*x)) + \
          5 / (1 + math.exp(-2*x)) - 2.5
    ret = max(.1, ret)
    return ret


def funct3(x):
    ret = max(0.01, (math.log(max(.001, 1 / (1 + math.exp(-100*x)) - 1 / (1 + math.exp(-2*x))) + x) + .5))
    return ret


def funct2(val):
    return max(.001, val + 1)
    


##################################
#
#10/17/2021
#Taking a more systematic approach.
#Build a lookup table from the data in Fig. 2.
#The preference table tells the proportion of ants choosing the E+F branch in
#two conditions, E+F vs. N, and E+F vs. E.
#This is converted to a lookup table that tells branch preference ratio as a function of
#amount of pheromone on each branch.   The conversion assumes an exponential
#decay in the amount of pheromone.
#


#frac_table_E
#fraction of ants choosing the E+F branch in the E condition, and N condition.
#These are the curves in Fig. 2
# each increment is 5 time steps.
#time axis           5    10   15   20   25   30   35   40   45   50
#                   55    60   65   70   75   80   85   90   95
gl_frac_table_e = [.94, .91, .83, .75, .68, .62, .56, .51, .52, .49, \
                   .50, .50, .50, .49, .51, .50, .50, .50, .51]  
gl_frac_table_n = [.94, .93, .89, .81, .78, .78, .80, .78, .76, .77, \
                   .73, .75, .74, .70, .66, .65, .54, .54, .48]




gl_decay = -.069       #Estimated from the data, how long it takes
                       #E+F vs. N to reach the pheromone noise level,
                       #starting from C1 = 10.
gl_noise_level = .02   #equivalent to minimum signal level, seat of the pants guess

 

#c1 is the assumed amount of pheromone on the E+F branch at the start
#c2 is the assumed amount of pheromone on the E branch at the start
#Decay of c1 to the noise level seems to occur after 90 steps.
#Call the noise level 0.02.
#this gives a decay rate of -.069
# 10 exp(90 * -.069) = 0.2
def setupTable(c1=10, c2=1):
    lut_ph1_ph2_frac = []  #(ph1, ph2, frac)
    c1 = c1 - gl_noise_level
    c2 = c2 - gl_noise_level
    for i in range(0, 19):
        frac_e = gl_frac_table_e[i]
        frac_n = gl_frac_table_n[i]
        t = (i+1)*5  #In Fig. 2, measurements start after 5 minutes
        ph1 = c1 * math.exp(gl_decay * t) + gl_noise_level
        ph2 = c2 * math.exp(gl_decay * t) + gl_noise_level
        print('t: ' + str(t) + ' frac_e: ' + str(frac_e) + ' frac_n: ' + str(frac_n) + ' ph1: ' + str(ph1) + ' ph2: ' + str(ph2))
        lut_ph1_ph2_frac.append((ph1, ph2, frac_e))
        lut_ph1_ph2_frac.append((ph1, gl_noise_level, frac_n))
    return lut_ph1_ph2_frac


def setTable():
    global gl_lut
    gl_lut = setupTable()

    
#gl_lut is a list of tuples: (pheromone1_level, pheromone2_level, preference_fraction)
try:
    gl_lut
except:
    gl_lut = setupTable()

    
def getLUT():
    return gl_lut


#plot fraction of E+F branch as a function of time using the lookup table data
#obtained from Fig. 2
#This assumes a single pheromone starting with concentrations 10, 1, and .02
#for the E+F, E, and N branches, respectively.
#This is not a parametric model.  This is directly using the data entered into the
#lookup table simply to re-compute the plots shown in the data.
#This just demonstrates that a single pheromone decay model can fit the data
#of Fig. 2. Thus, the claim in the paper,
# "For example, the observation that the initial frequency of ants choosing
#  the E+F branch is the same in the E+F vs N and E+F vs E experiments but
#  these frequencies diverge after 15 min (Fig. 2) is difficult to explain
#  with a single pheromone."
#is overstated.  In fact, a single pheromone model does the job.
def testExp2(c1=10, c2=1):
    duration = 120
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    br1_ar = []   #exponential decaying pheromone on branch 1
    br2a_ar = []  #exponential decaying pheromone on branch 2
    fbr1_ar = []  
    fbr2a_ar = []
    fbr2b_ar = []
    ratio2a_ar = []
    ratio2b_ar = []
                                       
    for m in range(duration):   #minutes
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

#find the entry in the lookup table closest to the ph1/ph2 fraction provided
#Since the data set is so small, just do a linear search to find the closest match.
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


#Plot the Lookup table of preference fraction versus measured pheromone on two branches.
#The measured pheremone is the exponentially decreasing amount of pheromone, but
#mapped through the measurement function amplifySignal() which nonlinearly amplifies
#small amounts.
#Preference fraction is the size of the circle.
def plotLUT(max_range=10):
    global gl_lut
    plt.figure(figsize=(6,6))
    min_v = 0
    max_v = 10
    x_ar = []
    y_ar = []
    val_ar = []
    for item in gl_lut:
        area = item[2] - .4
        val_ar.append(area*area*400)
        #print(str(area)) 
        x = item[0]
        y = item[1]
        x_amp = amplifySignal(x)
        y_amp = amplifySignal(y)
        #x_amp = x  #when not commented out, plot the raw amounts
        #y_amp = y
        x_ar.append(x_amp)
        y_ar.append(y_amp)
    plt.scatter(x_ar, y_ar, val_ar)
    plt.xlim([-.1, max_range])
    plt.ylim([-.1, max_range])
    plt.show()


#These parameters were estimated by plotting the observed data from Fig. 2
#and adjusting the parameters until the preference ratio transition occurred at
#about the same diagonal for the E+F vs. E condition and the E+F vs. N condition.
def amplifySignal(x):
    xpr = x - gl_noise_level
    val = 5.5*pow(xpr, .25)
    return val

#Returns distance to the midline of equal pheromone amounts.
#As everywhere, units are arbitrary pheromone amounts, assuming the
#E+F branch starts with pheromone amount 10.
def distToMid(xpr, ypr):
    xm = (xpr + ypr)/2
    ym = xm
    dist = pow(pow(xpr-xm, 2) + pow((ypr-ym), 2), .5)
    return dist


#These parameters were estimated from the plot of the observed data from Fig. 2.
#This squashing function creates a band of equal preference near the midline,
#transitioning to a preference for one branch or the other at a distance of
#about 2 from the midline.
def fracFromDistToMid(dist):
    frac = 1.0 / (1 + math.exp(-4*(dist-2))) - 1 / (1 + math.exp(8))    #ch1
    return frac


#The ratio map is a function of the distance to the midline fbr1 = fbr2
#and also the distance to the origin fbr1 = fbr2 = 0.
#There is a radius of about 8.
#Above this radius, the ratio can go up to .95
#Below this radius, the ratio is limited to about 0.6.
def mapGetRatio(fbr1, fbr2):
    dist_to_mid_2 = distToMid(fbr1, fbr2)
    dist_to_mid_ratio_factor = fracFromDistToMid(dist_to_mid_2)
    dist_to_orig = pow(pow(fbr1, 2) + pow((fbr2), 2), .5)
    dist_to_orig_ratio_factor = .6 + .4 * 1 / (1 + math.exp(-.8 * (dist_to_orig-8)))
    total_ratio = (dist_to_mid_ratio_factor * dist_to_orig_ratio_factor) / 2 + .5
    if fbr2 > fbr1:
        total_ratio = 1.0 - total_ratio
    return total_ratio



#Top level function.
#plot fraction of E+F branch as a function of time using a model of
#signal detection/amplification and then relative amounts of these,
#where the model form and parameters have been estimated in stages from Fig. 2
#The model is:
#  1. Amplify pheromone to get a measurement signal
#  2. Convert the measurement signals from the two branches to a preference ratio.
#     2a. Find distance from the signal amounts to the midpoint of the
#         equal-amounts line in measurement space.  Map using a squashing function.
#     2b. Find the distance from the signal amounts to the origin.
#         Map using a squashing function.
#     2c. Multiply the two considerations, distance to midpoint and distance to origin.
#This assumes a single pheromone starting with concentrations 10, 1, and .02
#for the E+F, E, and N branches, respectively.
def testExp4(c1=10, c2=1):
    c1 = c1 - gl_noise_level
    c2 = c2 - gl_noise_level
    duration = 120
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    br1_ar = []   #exponential decaying pheromone on branch 1
    br2a_ar = []  #exponential decaying pheromone on branch 2
    fbr1_ar = []  
    fbr2a_ar = []
    fbr2b_ar = []
    ratio2a_ar = []
    ratio2b_ar = []
    dist_a_ar = []
    dist_b_ar = []
                                       
    for m in range(duration):   #minutes
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
#
#Simulating Experiment 4, "dynamic environment" alternating food placement.
#

#We can use with the parameters used for Experiment 1-2, but the plot has
#some differences in appearance from Fig. 10.
#We can adjust the parameters to better fit Fig. 10, but then there is not
#as close a fit to Fig. 2.  Really, we would need more experimental data to
#find the right functional form and parameters to fit both figures well and
#predict other experimental results about ant behavior in this kind of setup.
#


#gl_ants_per_m = 50   #ch2
gl_ants_per_m = 30
gl_deposit_rate_exploit = .011    #pheromone per ant deposited
gl_deposit_rate_explore = .0011   #pheromone per ant deposited
gl_trial_duration = 45

#To simulate their experiment of placing food at one of the platforms at
#the end of a branch and watching activity ramp up and down, we need to
#model pheromone deposition as a function of whether the ant is exploring
#or exploiting.
#c1 and c2 are the initial pheromone levels at the left (1) and right (2) branch
#respectively.
#f1 and f2 are whether food is present at the left or right branch, respectively.
def simulateYJunctionTravel(c1 = 0, c2 = 0, f1=False, f2=False, ratio_ar=None):
    duration = gl_trial_duration
    ph1 = c1
    ph2 = c2
    print(f1)
    if ratio_ar == None:
        ratio_ar = []
    for m in range(duration):   #minutes
        frac_1 = mapGetRatio_Exp4(ph1, ph2)
        ratio_ar.append(frac_1)
        ants_1 = gl_ants_per_m * frac_1
        ants_2 = gl_ants_per_m * (1.0 - frac_1)
        #print('frac: ' + str(frac_1) + '  ants_1: ' + str(ants_1) + ' ants_2: ' + str(ants_2))
        ph1_next = ph1 * math.exp(gl_decay*1.2)
        ph2_next = ph2 * math.exp(gl_decay*1.2)
        if f1 == True:
            ph1_next += ants_1 * gl_deposit_rate_exploit
        else:
            ph1_next += ants_1 * gl_deposit_rate_explore
        if f2 == True:
            ph2_next += ants_2 * gl_deposit_rate_exploit
        else:
            ph2_next += ants_2 * gl_deposit_rate_explore
        #print('ph1_next: ' + str(ph1_next) + ' ph2_next: ' + str(ph2_next))
        ph1 = ph1_next
        ph2 = ph2_next
    return(ph1, ph2)



#The ratio map is a function of the distance to the midline fbr1 = fbr2
#and also the distance to the origin fbr1 = fbr2 = 0.
#There is a radius of about 8.
#Above this radius, the ratio can go up to .95
#Below this radius, the ratio is limited to about 0.6.
def mapGetRatio_Exp4(fbr1, fbr2):
    dist_to_mid_2 = distToMid(fbr1, fbr2)
    dist_to_mid_ratio_factor = fracFromDistToMid_Exp4(dist_to_mid_2)
    dist_to_orig = pow(pow(fbr1, 2) + pow((fbr2), 2), .5)
    dist_to_orig_ratio_factor = .8 + .2 * 1 / (1 + math.exp(-.8 * (dist_to_orig-8)))
    total_ratio = (dist_to_mid_ratio_factor * dist_to_orig_ratio_factor) / 2 + .5
    if fbr2 > fbr1:
        total_ratio = 1.0 - total_ratio
    return total_ratio


def fracFromDistToMid_Exp4(dist):
    #frac = 1.0 / (1 + math.exp(-4*(dist-1))) - 1 / (1 + math.exp(4))    #ch1
    frac = 1.0 / (1 + math.exp(-3.5*(dist-.8))) - 1 / (1 + math.exp(2.8))    #ch1
    return frac


#Top level function.
#This simulates the condition of Fig. 10 
#Food appears at the left branch for gl_trial_duration (45 minutes), then moves to
#the right branch for 45 minutes, then moves back to the left branch for 45 minutes.
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
    ax1.plot(ratio_ar)
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


##################################
#2021/10/18
#Experimenting with multi-component pheromones.
#The ant lays down a single pheromone during exploration or exploitation,
#but it contains two (for now) chemical components that evaporate at
#different rates.  The concentration ratio tells how long the pheromone
#has been there.
#
#Accurate measurement of a weak signal near the noise level really requires
#sampling and integrating the measurement over some period of time.

