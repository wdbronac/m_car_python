# -*- coding: utf-8 -*-
"""
Module tools_exp
"""

import numpy as np
import pylab as plt

def learn_episode_sarsa(env, agent):
    """
    The SARSA agent learns for one episode on the environment
    
    Parameters:
    ----------
    env : an instance of the cliff walking problem
        the system
    agent : an instance of sarsa 
        the agent
    
    Returns
    -------
    value : a float
        discounted sum of gathered rewards
    """
    cumR = 0 # cumulative rewards
    eoe = False
    state = env.observe()
    action = agent.sample_epsGreedy(state)
    while not eoe:
        reward, eoe = env.act(action)
        next_state = env.observe()
        next_action = agent.sample_epsGreedy(next_state)
        agent.update(state, action, reward, next_state, next_action, eoe)
        cumR += reward*agent.gamma
        state = next_state
        action = next_action
    return cumR
    
def learn_episode_qlearning(env, agent):
    """
    The Q-learning agent learns for one episode on the environment
    
    Parameters:
    ----------
    env : an instance of the cliff walking problem
        the system
    agent : an instance of qlearning
        the agent
    
    Returns
    -------
    value : a float
        discounted sum of gathered rewards
    """
    cumR = 0 # cumulative rewards
    eoe = False
    state = env.observe()
    while not eoe:
        action = agent.sample_epsGreedy(state)
        reward, eoe = env.act(action)
        next_state = env.observe()
        agent.update(state, action, reward, next_state, eoe)
        cumR += reward*agent.gamma
        state = next_state
    return cumR
    
def test_episode(env, agent):
    """
    The agent test the greedy policy of the learnt Q-function
    (without updating the parameters), for no more than 200 steps
    
    Parameters:
    ----------
    env : an instance of the cliff walking problem
        the system
    agent : an instance of san agent (either Q-learing or SARSA)
        the agent
    
    Returns
    -------
    value : a float
        discounted sum of gathered rewards
    set_states : a (T,2) array
        the set of states for the episode
    set_actions : a (T,) array
        the set of actions for the epsiode
    """
    lMax = 200 # do not generate a traj of more than lMax steps
    cumR = 0 # cumulative rewards
    eoe = False
    set_states = np.zeros((lMax, 2))
    set_actions = np.zeros(lMax)
    state = env.observe()
    action = agent.sample_greedy(state)
    cmpt = 0
    while (not(eoe) and (cmpt < lMax)): 
        reward, eoe = env.act(action)
        set_states[cmpt,:] = state
        set_actions[cmpt] = action
        next_state = env.observe()
        next_action = agent.sample_greedy(next_state)
        cumR += reward*agent.gamma
        state = next_state
        action = next_action
        cmpt +=1
    set_states = set_states[:cmpt,:]
    set_actions = set_actions[:cmpt]
    return cumR, set_states, set_actions
    
def plot_traj(set_states, set_actions, name, width = 12, height = 4):
    """"
    Plot a trajectory of the agent. The triangle marker indicates the kind of action,
    the position the state where the action has been chosen.
    
    Parameters
    ----------
    
    set_states : a (T,2) array
        the set of states for the episode
    set_actions : a (T,) array
        the set of actions for the epsiode
    name : a string
        name of the file for saving the fig
    width : an int
        the width of the problem (default: 12)
    height : an int
        the height of the problem (default: 12)
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim([-.5,width-.5])
    ax.set_ylim([-.5,height-.5])
    symb = ['>','<','^','v']
    for a in range(4):
        idx = np.where(set_actions==a)[0]
        ax.plot(set_states[idx,0], set_states[idx, 1], 'b'+symb[a], markersize=25)    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    fig.savefig(name+'.png')
    plt.close(fig)
    
def plot_res(res_sarsa, res_qlearning, name, smooth = True):
    """
    Plot the results. 
    
    Parameters
    ----------
    res_sarsa : a (T,) array
        the gathered rewards for each of the T episodes (training or test)
    res_qlearning : a (T,) array
        the gathered rewards for each of the T episodes (training or test)
    name : a string
        the name of the fig
    smooth : a boolean
        to plot a smoothed (more readable) figure (roughly, a low frequency filter is applied)
        
    """
    res_on = res_sarsa.copy()
    res_off = res_qlearning.copy()
    if smooth:
        alph = .01
        n = len(res_sarsa)
        for i in range(n-1):
            res_on[i+1] = (1-alph)*res_on[i] + alph*res_on[i+1]
            res_off[i+1] = (1-alph)*res_off[i] + alph*res_off[i+1] 
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(res_on, 'b', label='sarsa')
    ax.plot(res_off, 'r', label='qlearning')
    ax.legend()   
    ax.set_xlabel('number of episodes')
    ax.set_ylabel('gathered rewards')
    fig.savefig(name+'.png')
    plt.close(fig)
    
