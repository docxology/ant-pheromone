import matplotlib.pyplot as plt
import math

# ############################################3
# Assume a single pheromone that starts out at concentrations
# c1 and c2 on two branches respectively.
# Their amounts decay exponentially, or in one experiment,
# as the sum of two different exponentials.
# This is exploration of functional forms for the sensory response
# and then preference proportion.
#
# This section includes a lot of flopping around exploring functional forms.
# It was useful to use the Desmos online graphing calculator at
# https://www.desmos.com/calculator
#


def testExp1(c1, c2, decay=0.4):
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
    for m in range(duration):   # minutes
        br1 = c1 * (math.exp(-decay * m))
        br2a = c2 * (math.exp(-decay * m))
        br2b = 0
        # br1 = c1 * (math.exp(-decay * m) + .5 * math.exp(-decay * .5 * m))
        # br2a = c2 * (math.exp(-decay * m) + .5 * math.exp(-decay * .5 * m))
        # br2b = 0
        fbr1 = funct(br1)     # "E+F"
        fbr2a = funct(br2a)    # "E"
        fbr2b = funct(br2b)      # "N"
        diff12a = math.log(abs(fbr1 - fbr2a) + 1)
        diff12a = pow(diff12a, 1.5)
        diff12b = math.log(abs(fbr1 - fbr2b) + 1)
        diff12b = pow(diff12b, 1.5)
        # ratio2a = fbr1 / (fbr1 + fbr2a)
        ratio2a = (.2*diff12a + fbr1) / (fbr1 + fbr2a + diff12a)
        ratiom2a = ratioMap(ratio2a)
        # ratio2b = fbr1 / (fbr1 + fbr2b)
        ratio2b = (.2*diff12b + fbr1) / (fbr1 + fbr2b + diff12b)
        ratiom2b = ratioMap(ratio2b)
        if m < 10:
            print('fbr1: ' + str(fbr1) + ' fbr2a: ' + str(fbr2a) + ' ratio: '
                  + str(fbr1/fbr2a))
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
    # print('ratioMap x: ' + str(x) + ' ret: ' + str(ret))
    return max(0.0, min(1.0, ret))


def funct(x):
    # return math.log(x + 1)
    t1 = .5 * pow(x, .25) + .5 * x + .2
    # t1 = .5 * x + .2
    # t2 = 2 / (1 + math.exp(-2 * t1)) - 1
    # print('x: ' + str(x) + ' t1: ' + str(t1) + ' t2: ' + str(t2))
    # t2 = t1 + 1 / (1 + math.exp(-20*x)) - 1 / (1 + math.exp(-5*x))
    return t1


def funct2(x):
    return x + .01


def funct9(x):
    ret = 1 / (1 + math.exp(-20*x)) - \
        1 / (1 + math.exp(-5*x)) + x + .1
    return ret


def funct10(x):
    ret = 1 / (1 + math.exp(-40*x)) + 1*math.log(x+1.6)-1+0.25
    return ret


def funct1(x):
    ret = 1 / (1 + math.exp(-20*(x+.1))) + .1*x + .5
    return ret


def funct3(x):
    return math.log(x+1) + \
        .05 / (1 + math.exp(-50*x)) - \
        .05 / (1 + math.exp(-2*x)) + .05


def funct4(x):
    # ret = min(1, max(0.01, 1 / (1 + math.exp(-100*x)) - 1 /
    # (1 + math.exp(-2*x)) + .5))
    ret = 1 / (1 + math.exp(-100*x)) - \
          1 / (1 + math.exp(-2*x)) + \
          5 / (1 + math.exp(-2*x)) - 2.5
    ret = max(.1, ret)
    return ret


def funct5(x):
    ret = max(0.01, (math.log(max(.001, 1 / (1 + math.exp(-100*x)) - 1 /
                                  (1 + math.exp(-2*x))) + x) + .5))
    return ret


def funct6(val):
    return max(.001, val + 1)
