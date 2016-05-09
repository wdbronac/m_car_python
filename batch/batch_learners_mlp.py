# -*- coding: utf-8 -*-
"""
Module batch_learners_mlp
"""

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
class fittedQ:
    """
    The fittedQ algorithm
    
    Parameters
    ----------
    
    regressor : a regressor from scikit-learn
        the base regressor, default = extratree forest
    gamma : a real in (0,1)
        the discount factor, default = 0.99
    """
    
    def __init__(self, regressor=None, gamma = 0.99):
        self.na = 3 # set when calling the update function the first time
        self.nEpochs = 3000
	self.hidden_size = 8
        if regressor is None:
            regressor = Sequential()
            regressor.add(Dense(self.hidden_size, input_shape=(2,), activation='tanh', init='glorot_uniform')) #ici je triche un peu je sais qu'il y en a 2
            regressor.add(Dense(self.hidden_size, activation='tanh',  init='glorot_uniform'))
            regressor.add(Dense(3)) # je triche pcq je sais deja le nb d actions
            regressor.compile(sgd(lr=0.000001 ,momentum=0.9, nesterov=True), "mse")
        self.Q = regressor
        self.gamma = gamma
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
	#states = normalize(states)
	#print(states)	
        na = len(np.unique(actions))
        self.na = na
        qvals = np.zeros((n,na))
        print("fittedQ: iteration "+str(self.t))
        #X = np.concatenate((states, actions.reshape((n,1))), 1)
        #X = np.concatenate((states, actions.reshape((n,1))), 1)
        if self.t==0:
            Y = rewards
        else:
            for a in range(na):
                #qvals[:,a] = self.Q.predict(np.concatenate((next_states, a*np.ones((n,1))),1))
                qvals = self.Q.predict(normalize(next_states))
            Y = rewards + self.gamma*(1-eoes)*np.max(qvals, 1)
        # c est ici que je dois mettre le truc
	print('premiere normalisation')
        targets = self.Q.predict(normalize(states))
	#print("states.shape"+str(states.shape))
        for idx, t in enumerate(targets):
	    #print("idx: "+str(idx))
	    #print("t.shape" +str(t.shape))
	    #print("actions.shape" + str(actions.shape))
            t[actions[idx]] = Y[idx]
	    #print('X.shape'+str(X.shape))
	    #print('Targets.shape'+str(targets.shape))
	print('deuxieme normalisation')
	self.Q.fit(normalize(states), targets, nb_epoch=10, batch_size=32)
            #loss = self.Q.train_on_batch(X, targets)[0]
            #print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, win_cnt)) #bizarre cette ligne, au pire je mets un autre truc
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
        #qvals = np.zeros((n, self.na))
        #for a in range(self.na):
            #qvals[:,a] = self.Q.predict(np.concatenate((states, a*np.ones((n,1))),1))
	#states = normalize(states)
	qvals = self.Q.predict(normalize(states))
        gactions = np.argmax(qvals,1)
        values = qvals[(range(n), gactions)]
        return values, gactions
        
def normalize(state):
	state_input = np.copy(state)
        position = np.array([-1.2, 0.5])
        velocity = np.array([-0.07, 0.07])
	state_input[:,0] = (state_input[:,0]-position[0])/(0.25*(position[1]-position[0]))-2
	state_input[:,1] = (state_input[:,1]-velocity[0])/(0.25*(velocity[1]-velocity[0]))-2
	print(state_input)
	print('max_posi='+str(np.max(state_input[:,0])))
	print('max_velo='+str(np.max(state_input[:,1])))
	return state_input

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
        

