# -*- coding: utf-8 -*-
"""
Module plot_tools
"""

import pylab as plt
import numpy as np

def plot_transitions(states, actions, next_states, name):
    """
    Plot the transitions, different colors for different actions
    (blue for left, green for none, red for right). Transitions in bold indicate
    an informative reward.
    
    Parameters
    ----------
        states : a (n,2) array
            the set of states
        actions : a (n,) array
            the set of actions
        next_states : a (n,2) array
            the set of next states
        name: a string
            the name of the file for saving the fig
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    clr = ['b', 'g', 'r'] # left: red, none: black, g: right
    (n,ds) = np.shape(states)
    for i in xrange(n):
        if next_states[i,0]==.5 or next_states[i,0]==-1.2:
            ax.plot([states[i,0], next_states[i,0]],[states[i,1], next_states[i,1]], color=clr[actions[i]], linewidth='5')
        else:
            ax.plot([states[i,0], next_states[i,0]],[states[i,1], next_states[i,1]], color=clr[actions[i]])
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    fig.savefig(name+'.png')
    plt.close(fig)
    
def plot_val_pol(Q, name):
    """
    
    Plot the value function (max_a Q(s,a)) and the greedy policy for a Q-function.
    
    For the value function, red corresponds to high values and blue to low ones.
    
    For the policy, red corresponds to going right, blue to left and green to none.
    
    Parameters
    ----------
    Q : a Q-function
        should handle Q.predict
    name : a string
        the name of the file to be saved
    """
    mesh_grain_val = 100
    mesh_grain_pol = 30
    # plot the resulting value function and policy
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121)
    pos = np.arange(-1.2, 0.5, 1.7/mesh_grain_val)
    vel = np.arange(-.07, .07, .14/mesh_grain_val)
    pos, vel = np.meshgrid(pos, vel)
    V, A = Q.predict(np.c_[pos.ravel(), vel.ravel()])
    V = V.reshape(pos.shape)
    ax.contourf(pos, vel, V,100)
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    ax = fig.add_subplot(122)
    pos = np.arange(-1.2, 0.5, 1.7/mesh_grain_pol)
    vel = np.arange(-.07, .07, .14/mesh_grain_pol)
    pos, vel = np.meshgrid(pos, vel)
    V, A = Q.predict(np.c_[pos.ravel(), vel.ravel()])
    A = A.reshape(pos.shape)
    ax.contourf(pos, vel, A)
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    fig.savefig(name+'.png')
    plt.close(fig)
    
def plot_traj(traj, name):
    """
    
    Plot the trajectory.
    
    Parameters
    ----------
    
    traj : a (T,2) array
        the set of T states of a trajectory
    name : a string
        the name of the file to be saved
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim([-1.2,0.5])
    ax.set_ylim([-.07,.07])    
    ax.plot(traj[:,0], traj[:,1])
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    fig.savefig(name+'.png')
    plt.close(fig)
    
def plot_perf(res, name):
    """
    
    Plot the performance.
    
    Parameters
    ----------
    res : a (T,) array
        res[i] should corresponds to the number of time step required to go out of the mountain, for 
        the greedy policy learnt after i iterations
    name : a string
        the name of the file to be saved
    
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)   
    ax.plot(res)
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('performance')
    fig.savefig(name+'.png')
    plt.close(fig)
