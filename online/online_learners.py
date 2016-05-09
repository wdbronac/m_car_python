# -*- coding: utf-8 -*-
"""
Module online_learners
"""

import numpy as np

class SARSA:
    """
    A SARSA agent, specifically for the cliff walking problem with a tabular 
    representation of the Q-function.
    
    Parameters
    ----------
    width : an int
        the width of the cliff problem (default: 12)
    height : an int
        the height of the cliff problem (default: 4)
    gamma : a float in (0,1) 
        the discount factor (default: 0.99)
    epsilon : a float in (0,1)
        the exploration factor, for epsilon-greedy (default: 0.1)
    alpha : a float in (0,1)
        the learning factor (default: 0.1)
    """
    
    def __init__(self, width=12, height=4, gamma = 0.99, epsilon = 0.1, alpha = 0.1):
        # set the values
        self.w = width 
        self.h = height
        self.na = 4 # number of actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        
        # init the Q-function
        self.Q = np.ones((self.w*self.h, self.na))
        
    def update(self, state, action, reward, next_state, next_action, eoe):
        """"
        Update the Q-values given a provided (s,a,r,s',a') transition.
        
        Parameters
        ----------
        state : a (2,) array
            the state
        action : an int
            the action
        reward : a float
            the reward
        next_state : a (2,) array
            the new state
        next_action : an int
            the new action
        """
        # transform the state into index
        i = state[0]*self.h + state[1]
        j = next_state[0]*self.h + next_state[1]
        self.Q[i, action] = self.Q[i, action] + \
        self.alpha*(reward + self.gamma*(1-eoe)*self.Q[j, next_action] -self.Q[i, action])
        
    def sample_greedy(self, state):
        """
        Choose the greedy action.
        
        Parameters
        ----------
        state : a (2,) array
            the state
        
        Returns
        -------
        action : an int
            the greedy action
        """
        i = state[0]*self.h + state[1]
        return np.argmax(self.Q[i,:])
    
    def sample_epsGreedy(self, state):
        """
        Choose an action according to an epsilon greedy policy.
        
        Parameters
        ----------
        state : a (2,) array
            the state
            
        Returns
        -------
        action : an int
            the sampled action
        """
        if np.random.rand()<self.epsilon:
            return np.random.randint(0,self.na)
        else:
            i = state[0]*self.h + state[1]
            return np.argmax(self.Q[i,:])
            
            
class Qlearning:
    """
    A Q-learning agent, specifically for the cliff walking problem with a tabular 
    representation of the Q-function.
    
    Parameters
    ----------
    width : an int
        the width of the cliff problem (default: 12)
    height : an int
        the height of the cliff problem (default: 4)
    gamma : a float in (0,1) 
        the discount factor (default: 0.99)
    epsilon : a float in (0,1)
        the exploration factor, for epsilon-greedy (default: 0.1)
    alpha : a float in (0,1)
        the learning factor (default: 0.1)
    """
    
    def __init__(self, width=12, height=4, gamma = 0.99, epsilon = 0.1, alpha = 0.1):
        # set the values
        self.w = width 
        self.h = height
        self.na = 4 # number of actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        
        # init the Q-function
        self.Q = np.zeros((self.w*self.h, self.na))
        
    def update(self, state, action, reward, next_state, eoe):
        """"
        Update the Q-values given a provided (s,a,r,s') transition.
        
        Parameters
        ----------
        state : a (2,) array
            the state
        action : an int
            the action
        reward : a float
            the reward
        next_state : a (2,) array
            the new state
        next_action : an int
            the new action
        """
        # transform the state into index
        i = state[0]*self.h + state[1]
        j = next_state[0]*self.h + next_state[1]
        self.Q[i, action] = self.Q[i, action] + \
        self.alpha*(reward + self.gamma*(1-eoe)*np.max(self.Q[j, :]) -self.Q[i, action])
        
    def sample_greedy(self, state):
        """
        Choose the greedy action.
        
        Parameters
        ----------
        state : a (2,) array
            the state
        
        Returns
        -------
        action : an int
            the greedy action
        """
        i = state[0]*self.h + state[1]
        return np.argmax(self.Q[i,:])
    
    def sample_epsGreedy(self, state):
        """
        Choose an action according to an epsilon greedy policy
        
        Parameters
        ----------
        state : a (2,) array
            the state
            
        Returns
        -------
        action : an int
            the sampled action
        """
        if np.random.rand()<self.epsilon:
            return np.random.randint(0,self.na)
        else:
            i = state[0]*self.h + state[1]
            return np.argmax(self.Q[i,:])