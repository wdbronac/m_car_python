# -*- coding: utf-8 -*-
"""
Module batch_learners
"""

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

class fittedQ:
    """
    The fittedQ algorithm
    
    Parameters
    ----------
    
    regressor : a regressor from scikit-learn
        the base regressor, default = extra-tree forest
    gamma : a real in (0,1)
        the discount factor, default = 0.99
    """
    
    def __init__(self, regressor=None, gamma = 0.99):
        if regressor is None:
            regressor = ExtraTreesRegressor(n_estimators = 50, min_samples_leaf = 5)
            #regressor = LinearRegression()
        self.Q = regressor
        self.gamma = gamma
        self.na = 0 # set when calling the update function the first time
        self.t = 0 # current iteration
            
    def update(self, states, actions, next_states, rewards, eoes):
        """
        Perform just one update of fitted-Q. 
        
        Parameters
        ----------
        
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
        (n,d) = np.shape(states)
        na = len(np.unique(actions))
        self.na = na
        qvals = np.zeros((n,na))
        print("fittedQ: iteration "+str(self.t))
        X = np.concatenate((states, actions.reshape((n,1))), 1)
        if self.t==0:
            Y = rewards
        else:
            for a in range(na):
                qvals[:,a] = self.Q.predict(np.concatenate((next_states, a*np.ones((n,1))),1))
            Y = rewards + self.gamma*(1-eoes)*np.max(qvals, 1)
        self.Q.fit(X, Y)
        self.t +=1
            
    def predict(self, states):
        """
        Predict values (max_a Q(s,a)) and greedy actions for given states
        
        Parameters
        ----------
        
        states : a (n,) array
            the set of states
            
        Returns
        -------
        
        values : a (n,) array
            the values
        
        gactions : a (n,) array
            greedy actions
        """
        (n,d) = np.shape(states)
        qvals = np.zeros((n, self.na))
        for a in range(self.na):
            qvals[:,a] = self.Q.predict(np.concatenate((states, a*np.ones((n,1))),1))
        gactions = np.argmax(qvals,1)
        values = qvals[(range(n), gactions)]
        return values, gactions
        

class LSPI:
    """
    The LSPI algorithm, with hand coded features
    
    Parameters
    ----------
    gamma : a float in (0,1)
        the discount factor (default: 0.99)
    """
    
    def __init__(self, gamma = 0.99):
        self.nf = 3 #number of Gaussians per dim
        self.na = 3 #number of actions, need it
        self.d = (self.nf*self.nf+1)
        self.theta = np.zeros(self.d*self.na) # param vector        
        self.t = 0 # current iteration
        self.gamma = gamma
        
    def features(self, states, actions):
        """
        Return the set of features for a given set of state-action couples
        
        Parameters
        ----------
        
        states : a (n,2) array
            the set of states
        actions : a (n,) array
            the actions
            
        Returns
        -------
        
        features : a (n,d) array
            the set of features
        """
        pmin = -1.2
        pmax = .5
        vmin = -.07
        vmax = .07
        sigmap = (pmax-pmin)/(self.nf-1)
        sigmav = (vmax-vmin)/(self.nf-1)
        
        na = self.na
        (n,ds) = np.shape(states)
        d = self.d # number of features per action
        features = np.zeros((n, na*d))
        for i in xrange(n):
            features[i, (actions[i]+1)*d-1]=1 # last feature is constant
            for jp in xrange(self.nf):
                for jv in xrange(self.nf):
                    cp = (states[i,0] - (pmin + jp*sigmap))/sigmap
                    cv = (states[i,1] - (vmin + jv*sigmav))/sigmav
                    features[i, actions[i]*d + jp + jv*self.nf] = np.exp(-.5*(cp*cp + cv*cv))
        return features
                    
    def update(self, states, actions, next_states, rewards, eoes):
        """
        Perform just one update of LSPI.
        
        Parameters
        ----------
        
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
        (n,ds) = np.shape(states)
        print("LSPI: iteration "+str(self.t))
        # current policy for the dataset (greed. resp to Q_theta)
        gactions = np.zeros(n) #np.random.randint(0,3,n)
        if self.t>0:
            qvals = self.get_qvals(next_states)
            gactions = np.argmax(qvals, 1)
        phi = self.features(states, actions)
        next_phi = self.features(next_states, gactions)
        for i in xrange(n):
            if eoes[i]:
                next_phi[i,:] = 0
        A = np.dot(np.transpose(phi), phi - self.gamma * next_phi) 
        b = np.dot(np.transpose(phi), rewards)
        self.theta = np.linalg.solve(A, b)
        self.t +=1
            
    def get_qvals(self, states):
        """
        Get the Q values for a set of states
        
        Parameters
        ----------
        
        states : a (n,2) array
            the set of states
            
        Returns
        -------
        
        qvals : a (n,3) array
            the set of Q-values (for the provided states and each action)
        """        
        (n,ds) = np.shape(states)
        qvals = np.zeros((n, self.na))
        for a in range(self.na):
            qvals[:,a] = np.dot(self.features(states, a*np.ones(n)), self.theta)
        return qvals
        
    def predict(self, states):
        """
        Predict values (max_a Q(s,a)) and greedy actions for given states
        
        Parameters
        ----------
        
        states : a (n,2) array
            the set of states
            
        Returns
        -------
        
        values : a (n,) array
            the values
        
        gactions : a (n,) array
            greedy actions
        """
        (n,d) = np.shape(states)
        qvals = self.get_qvals(states)
        gactions = np.argmax(qvals,1)
        values = qvals[(range(n), gactions)]
        return values, gactions
        

