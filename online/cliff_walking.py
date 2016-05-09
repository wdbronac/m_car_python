# -*- coding: utf-8 -*-
"""
Module cliff_walking
"""

import numpy as np

class simulator:
    """
    The cliff walking problem.
    
    Parameters
    ----------
    width : an int
        the width of the problem (default: 12)
    heigth : an int
        the height of the problem (default: 4)
    """
    
    def __init__(self, width = 12, height = 4):
        self.w = width
        self.h = height
        # the intial state is always (0,0), I do not allow to sample from any state
        self.state = np.array([0, 0]) 

    def observe(self):
        """
        Get the current state of the system.
        
        Returns:
        --------
        s : a (2,) array
            the current state
        """
        return self.state.copy()
    
    def act(self, action):
        """
        Transition in the simulator.
        
        Parameters
        ----------
        action : an int in (0,3) 
            the action (right/left/up/down)
        
        Returns
        -------
        reward : a real
            the reward
        eoe : a boolean
            a flag, set to true if the episode is ended
        """
        # new state
        if action==1:                                   # go left
            self.state[0] = np.max([self.state[0]-1, 0])
        elif action==0:                                 # go right
            self.state[0] = np.min([self.state[0]+1, self.w-1])
        elif action==2:                                 # go up
            self.state[1] = np.min([self.state[1]+1, self.h-1])
        else:                                           # go down
            self.state[1] = np.max([self.state[1]-1, 0])
        # reward and eoe
        reward = -1
        eoe = False
        if self.state[1]==0:                            # first raw
            if self.state[0]>0:                         # not the start state
                if self.state[0]==self.w-1:             # goal state
                    reward = 0
                    eoe = True
                    self.state = np.array([0,0])
                else:                                   # the cliff !
                    reward = -100
                    eoe = True
                    self.state = np.array([0,0])
        return reward, eoe                    
