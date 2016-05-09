# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:59:47 2015

@author: geist_mat
"""

import numpy as np
import pylab as plt
from sklearn.ensemble import ExtraTreesRegressor

import batch_learners_simple as bl
#import batch_learners as bl
import mountain_car as mc
import plot_tools as pt
reload(mc)
reload(bl)
reload(pt)
plt.close("all")
gamma = 0.99 # discount factor
lmax = 300 # maximum length of a trajectory

# create the simulator
sim = mc.simulator(lmax)



# sample transitions (to plot, not too much)
n = 1500
(states, actions, next_states, rewards, eoes) = sim.gen_dataset(n)
pt.plot_transitions(states, actions, next_states, 'data_set')


# create a dataset (bigger) for learning
n = 30000
(states, actions, next_states, rewards, eoes) = sim.gen_dataset(n)

# fitted-Q
################
# create the learner and learn (fitted-Q)

reg = ExtraTreesRegressor(n_estimators = 50, min_samples_leaf = 5)
T = 150
fq = bl.fittedQ()#regressor = reg, gamma = gamma)
res = np.zeros(T)
for t in range(T):
    fq.update(states, actions, next_states, rewards, eoes)
    name = './figs/fq_val_pol_%03d.png'%t
    pt.plot_val_pol(fq, name)
    traj, l = sim.sample_traj(fq)
    print("       length sampled greedy traj.: "+str(l))
    name = './figs/fq_traj_%03d'%t
    pt.plot_traj(traj, name)
    res[t] = l
pt.plot_perf(res, 'fq_perfs')


# LSPI
###############
T = 5
lspi = bl.LSPI(gamma)
res = np.zeros(T)
for t in range(T):
    lspi.update(states, actions, next_states, rewards, eoes)
    name = './figs/lspi_val_pol_%03d.png'%t
    pt.plot_val_pol(lspi, name)
    traj, l = sim.sample_traj(lspi)
    print("       length sampled greedy traj.: "+str(l))
    name = './figs/lspi_traj_%03d'%t
    pt.plot_traj(traj, name)
    res[t] = l
pt.plot_perf(res, 'lspi_perfs')

plt.close("all")
