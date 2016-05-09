# -*- coding: utf-8 -*-
"""
Module mountain_car
"""

import numpy as np

class simulator:
    """
    The mountain-car simulator, as described by Sutton & Barto.
    The state space is [-1.2, 0.6]x[-0.07, 0.07] (position/velocity).
    The action set is (0,2) (0 is left, 1 is none, 2 is right).
    The reward is 0 for escaping, -1 else.
    
    Parameter
    ---------
        max_step : an int
            the maximum number of steps used for generating a trajectory (default: 300)
    """
    
    def __init__(self, max_step = 300):
        self.max_step = max_step
        self.position = np.array([-1.2, 0.5])
        self.velocity = np.array([-0.07, 0.07])
        
    def transition(self, state, action):
        """
        The transition function
        
        Parameters
        ----------
            state : a (2,) array
                starting state
            action : an int in (0,2)
                chosen action
            
        Returns
        -------
            next_state : a (2,) array
                resulting state
            r : a scalar
                reward associated to the transition
            eoe : boolean
                end of episode flag (True if episode is ended, False else)
            
        """
        next_state = np.zeros(2)
        eoe = False
        r = -0.05
        next_state[1] = state[1] + 0.001*(action-1) -0.0025*np.cos(3*state[0])
        next_state[0] = state[0] + next_state[1]
        if next_state[0] > self.position[1]:
            r = 0.5
            eoe = True
        if next_state[0] < self.position[0]:
            next_state[1] = 0
            r = -0.5
        next_state[0] = min(max(next_state[0], self.position[0]), self.position[1])        
        next_state[1] = min(max(next_state[1], self.velocity[0]), self.velocity[1])
        return next_state, r, eoe
        
    def gen_dataset(self, n):
        """
        Generate a dataset of sampled transitions. The initial state is chosen 
        uniformly at random, the action is chosen uniformly at random and the next 
        state (and end of episode) are sampled using the dynamics.
        
        Parameters
        ---------
            n : n int
                the number of transitions
        
        Returns
        -------
            states : a (n,2) array
                the set of states
            actions : a (n,) array
                the set of actions
            next_states : a (n,2) array
                the set of next states
            rewards : a (n,) array
                the set of gathered rewards
            eoes : a (n,) array
                the set of flags for episode ending
        """
        
        # init
        states = np.zeros((n, 2))
        next_states = np.zeros((n,2))
        rewards = np.zeros(n)
        eoes = np.zeros(n)
        
        # initial random states
        states[:,0] = self.position[0] + (self.position[1] - self.position[0])*np.random.random_sample(n)
        states[:,1] = self.velocity[0] + (self.velocity[1] - self.velocity[0])*np.random.random_sample(n)        
        # random actions
        actions = np.random.randint(0, 3, n)
        # random transitios
        for i in xrange(n):
            (next_states[i,:], rewards[i], eoes[i]) = self.transition(states[i,:], actions[i])
        return states, actions, next_states, rewards, eoes
        
    def sample_traj(self, Q):
        """
        Sample a trajectory in the mountain car problem,
        using a given Q-function, for a maximum of max_step.
        The initial state is the equilibrium position (-pi/6, 0).
        
        Parameters
        ----------
        Q : a Q-function
            should handle Q.predict
            
        
        Returns
        -------
        trajectory : a (n,2) array
            the sampled trajectory
            
        lt : an int
            the length of the trajectory
        """
        cmpt = 0
        trajectory = np.zeros((self.max_step,2))
        eoe = False
        #state = np.array([-np.pi/6 - .17/2 + .17*np.random.rand(),0 - .007 + .014*np.random.rand()])
        state = np.array([-np.pi/6 ,0 ])
        while ((eoe==False) & (cmpt<self.max_step)):
            trajectory[cmpt,:] = state
            v, action = Q.predict(state.reshape((1,2)))
            (next_state, r, eoe) = self.transition(state, action)
            state = next_state
            cmpt +=1
        trajectory = trajectory[:cmpt,:]
        return trajectory, cmpt
        
    
