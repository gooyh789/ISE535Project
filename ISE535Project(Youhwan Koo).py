# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:49:15 2021

@author: gooyh
"""

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

# Missile data(Mdata)
path = r"C:\Users\gooyh\Desktop\개인파일\NCSU\2021 Fall\ISE 535 Python Programming in ISE\Project\\"
Mdata = pd.read_csv(path + "north_korea_missile_tests_database_050217.csv")
date = Mdata['Date']

# Organizing new dataset that have year information
year = np.zeros(157,)
for i in range(len(date)):
    new = datetime.strptime(date[i], '%d-%b-%y').year
    year[i] = new
print(year)

# Check the frequency of year in the dataset
yearSet = {}
yearList = np.unique(year)
for val in year:
    if val in yearSet:
        yearSet[val] += 1
    else:
        yearSet[val] = 1    
print(yearSet)

Y = []
T = []
for key, values in yearSet.items():
    Y.append(key)
    T.append(values)    

# Plot the histogram to look the trend
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.hist(year, facecolor='r', alpha=0.75, bins=33)
plt.plot(Y,T, color='red', linewidth=2)
plt.title("North Korea Missile Test History", fontsize=15, fontweight="bold")
plt.ylabel("Number of launch test")
plt.xlabel("Year")
plt.grid(axis="x")
plt.show()

# Build first simple M/M/1 queue model
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Initializae
lda = 10 # average detection time
mu1 = 2 # average time to active SK missile in upper layer 
mu2 = 3 # average time to active SK missile in lower layer
eps = 1 # average time of NK missile explode in SK territory
k = 108 # number of missile that can be entered in the system
 
def model(lda, mu1, mu2, eps, k):
    trm = np.zeros(shape=(k+1,k+1))
    for i in range(len(trm)):
        if i < k:
            trm[i,i+1] = lda
        if i > 0:
            trm[i,i-1] = i*(mu1+2*mu2+eps)

    # intensity list
    intensity = [1]
    for i in range(1,len(trm)):
        val = intensity[-1]*(lda/trm[i,i-1])
        intensity.append(val)
    s0 = 1/sum(intensity)

    # Steady state probability
    steady_state = [s0]
    for i in range(1,len(trm)):
        steady_state.append(s0*intensity[i])

    return steady_state

model(lda=10, mu1=2, mu2=3, eps=1, k=108)

# Design simulation Using simpy
import simpy
lda = 10 # average detection rate (average interarrival detection time: 2min)
mu1 = 2 # average rate to active SK missile in upper layer (average SK missile active in upper: 6min) 
mu2 = 3 # average rate to active SK missile in lower layer (average SK missile active in lower: 5min)
eps = 1 # average rate of NK missile explode in SK territory (average NK missile active: 10min)

result = {'success':0, 'fail':0}
def operation_process(env, enter_time):
    while True:
        yield env.timeout(enter_time)
        print(f"Missile is entering at {env.now}")
        attack_duration = -math.log(np.random.rand(1))/(eps)
        operation_duration = -math.log(np.random.rand(1))/(mu1+2*mu2)
        if attack_duration <= operation_duration:
            result['fail'] += 1
            yield env.timeout(attack_duration)
            print(f"Mission fail at {env.now}")
        else:
            u = np.random.rand(1)
            if u <= 0.7:
                result['success'] += 1
                yield env.timeout(operation_duration)
            else:
                result['fail'] += 1
                yield env.timeout(operation_duration)
                print(f"Mission fail at {env.now}")
            print(f"Mission success at {env.now}")

enter_time = 1
env = simpy.Environment() 
env.process(operation_process(env, enter_time))
env.run(until = 3000)

prob_success = result['success'] /(result['success'] + result['fail'])
print("Probability of success:", prob_success)

# Design the simple simulation
def maxRate(lda, mu1, mu2, eps, k):
    rate = k*(mu1+2*mu2+eps) + lda
    return rate

# Probability of new missile launch
def getP1(Mrate, lda, n, k): 

    if n == k:
        p1 = 0
    else:
        p1 = lda / Mrate
    return(p1)

# Probability of missile performance
def getP2(Mrate, mu1, mu2, eps, n, lower, upper):  
    p2 = n*(upper*mu1 + lower*mu2 + eps) / Mrate
    return(p2)

# Probability of SK missile meet before NK missile blasting
def getP3(mu1, mu2, eps, n, lower, upper):  
    p3 = (upper*mu1+lower*mu2) / (upper*mu1+lower*mu2+eps)
    return(p3)

lda = 10 # average detection rate (average interarrival detection time: 0.1min)
mu1 = 2 # average rate to active SK missile in upper layer (average SK missile active in upper: 0.5min) 
mu2 = 3 # average rate to active SK missile in lower layer (average SK missile active in lower: 0.3min)
eps = 1 # average rate of NK missile explode in SK territory (average NK missile active: 1min)
k = 108 # number of missile in the system
lower = 2
upper = 1
n = 0
tE = 0
t = 0
NA = 0 # cumulative number of detected missile
ND = 0 # cumulative number of missile operation result
NT = [] # number of missile set in the system
AT = [] # detected time set
DT = [] # result time set
LT = 100 # time limit in long run
S = 0
F = 0
Mrate = maxRate(lda, mu1, mu2, eps, k)

while t < LT:
    tE = -math.log(np.random.rand(1))/Mrate
    t = t + tE
    U = np.random.rand(1)
    p1 = getP1(Mrate, lda, n, k)
    p2 = getP2(Mrate, mu1, mu2, eps, n, lower, upper)
    if U <= p1:
        n = n+1
        NA = NA+1
        AT.append(t)
        NT.append(n)
    elif U <= (p1+p2):
        V = np.random.rand(1)
        p3 = getP3(mu1, mu2, eps, n, lower, upper)
        if V <= p3:
            S = S+1
        else:
            F = F+1
        n = n-1
        ND = ND+1
        DT.append(t)
        NT.append(n)
    else:
        n = n
        NT.append(n)        
    print(f"Simulation processing...{(t/LT)*100}%")

# Proportion of operation execution and terrorist act
print(S / (S+F+NA)) # operation execution
print(F / (S+F+NA)) # terrorist act

state = {}
for val in NT:
    if val in state:
        state[val] += 1
    else:
        state[val] = 1
sumVal = sum(state.values())
for key, values in state.items():
    state[key] = values/sumVal

state = pd.DataFrame(state.items(), columns = ['State', 'Probability']).sort_values(['State'])
print(state)


# Design the more complicated simulation (adding constraints)
# NK's total missile number : NK
# SK's total missile number : SK
# Probability of destroy NK missile in upper layer: 0.8
# Probability of destroy NK missile in lower layer: 0.7
# Detection capability of SK surveillance system: w

def getP1(Mrate, lda, n, k, NA, NK): 
    if NA < NK:
        if n == k:
            p1 = 0
        else:
            p1 = lda / Mrate
    else:
        p1 = 0
    return(p1)

# Probability of missile performance (modifed for constraints)
def getP2(Mrate, mu1, mu2, eps, n, lower, upper, south_missile, SK):  
    if south_missile < SK:
        p2 = n*(upper*mu1 + lower*mu2 + eps) / Mrate
    else:
        p2 = n*eps / Mrate
    return(p2)

# Probability of SK missile meet before NK missile blasting (modifed for constraints)
def getP3(mu1, mu2, eps, n, lower, upper, south_missile, SK):  
    if south_missile < SK:
        p3 = (upper*mu1+lower*mu2) / (upper*mu1+lower*mu2+eps)
    else: 
        p3 = 0  # If south korea has no missile, probability become 0
    return(p3)

# Complex simulation model adding new constraints
lda = 10 # average detection rate (average interarrival detection time: 2min)
mu1 = 2 # average rate to active SK missile in upper layer (average SK missile active in upper: 6min) 
mu2 = 3 # average rate to active SK missile in lower layer (average SK missile active in lower: 5min)
eps = 1 # average rate of NK missile explode in SK territory (average NK missile active: 10min)
k = 108 # number of missile in the system
lower = 2
upper = 1
Mrate = maxRate(lda, mu1, mu2, eps, k)
upper_prob = 0.8
lower_prob = 0.7
SC = 1 # Surveillance capability (Detection capability)
numLong = 450
numShort = 500
NK = numLong+numShort
SK = 2000
probList = []
up_low_List = []
simul = 100
for i in range(simul):
    S = 0
    F = 0
    missing = 0
    t = 0
    n = 0
    NA = 0 # cumulative number of detected missile
    ND = 0 # cumulative number of missile operation result
    NT = [] # number of missile set in the system
    AT = [] # detected time set
    DT = [] # result time set
    upper_missile = 0
    up_to_low_missile = 0
    lower_missile = 0
    south_missile = 0
    nk_quantity = NK
    while ND != NK:
        w = np.random.rand(1)
        if w > SC:
            missing += 1
            NK -= 1
        else:
            tE = -math.log(np.random.rand(1))/Mrate
            t = t + tE
            U = np.random.rand(1)
            p1 = getP1(Mrate, lda, n, k, NA, NK)
            p2 = getP2(Mrate, mu1, mu2, eps, n, lower, upper, south_missile, SK)
            if U <= p1:
                n = n+1
                NA = NA+1
                if south_missile < SK:
                    south_missile += (lower+upper)
                    if south_missile > SK:
                        south_missile = SK
                AT.append(t)
                NT.append(n)
            elif U <= (p1+p2):
                V = np.random.rand(1)
                p3 = getP3(mu1, mu2, eps, n, lower, upper, south_missile, SK)
                if p3 == 0:
                    F += nk_quantity
                    break # If there is no missile 
                else:
                    if V < p3: # detected missile enters in upper layer
                        rv_missile = np.random.rand(1)
                        if rv_missile <= numLong / (numLong + numShort): # operating missile is in the upper
                            upper_missile += 1
                            s = np.random.rand(1)
                            if NA < 48: # The number of SK upper layer missile
                                if s <= upper_prob:
                                    S += 1
                                    n = n-1
                                    nk_quantity -= 1
                                    ND = ND+1
                                    DT.append(t)
                                    NT.append(n)
                                else:
                                    up_to_low_missile += 1
                                    n = n
                                    NT.append(n)
                            else:
                                lower_missile += 1
                                s = np.random.rand(1)
                                if s <= lower_prob:
                                    S += 1
                                else:
                                    F += 1
                                n = n-1
                                nk_quantity -= 1
                                ND = ND+1
                                DT.append(t)
                                NT.append(n) 
                        else: # detected missile enters in lower layer already
                            lower_missile += 1
                            s = np.random.rand(1)
                            if s <= lower_prob:
                                S += 1
                            else:
                                F += 1
                            n = n-1
                            nk_quantity -= 1
                            ND = ND+1
                            DT.append(t)
                            NT.append(n)                    
                    else:
                        rv_missile = np.random.rand(1)
                        if rv_missile <= numLong / (numLong + numShort): # operating missile is in the upper
                            upper_missile += 1
                            F += 1
                        else:
                            lower_missile += 1
                            F += 1
                        n = n-1
                        nk_quantity -= 1
                        ND = ND+1
                        DT.append(t)
                        NT.append(n)
            else:
                n = n
                NT.append(n) 
    probList.append(S/(S+F))
    print(f"Simulation processing...{((i+1)/simul)*100}%")

# 95% CI for success proportion and the number of missiles (upper to lower)
meanProp = np.mean(probList) # average proportion
print(round(meanProp,3)) # average proportion
Prop_95CI = [round(meanProp - 1.96*math.sqrt(np.var(probList)/simul),3), 
             round(meanProp + 1.96*math.sqrt(np.var(probList)/simul),3)]
print(Prop_95CI)

# Arrival and departure graph
ET = AT + DT
ET.sort()
N = []
n = 0
for i in ET:
    if i in AT:
        n += 1
        N.append(n)
    else:
        n -= 1
        N.append(n)
    
plt.style.use("ggplot")
plt.title("North Korea Missile Track", fontsize=15, fontweight="bold")
plt.ylabel("The number of missile in the system")
plt.xlabel("Time")
plt.step(ET, N, where="post")
plt.show() 

