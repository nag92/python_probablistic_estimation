'''
Created on Dec 31, 2015

@author: nathaniel
This is a particle filter for a single variable
uses lamnda functions for sensors and model 

'''

import numpy as np
import scipy.stats
from numpy import cumsum
import random
from scipy.signal.signaltools import resample
from numpy.dual import inv

class Particle_Filter(object):
    
    def __init__(self,x , model,sensor,m,dt,varence):
        self.model = model #lamnda function for model
        self.M  = m #amount of particles
        self.sensor = sensor #lamnda function for sensor
        self.dt = dt # time step
        self.varence = varence #error
        self.pred_x = [] #holds the particles
        self.time = 0 #time
        
       
        self.pred_x = np.random.normal(x,varence**2,self.M)
        
    def update(self, u,z ):
        #update the particles based on model
        x_update = [self.model(x,u,self.time) for x in self.pred_x]
        #update the sensors positions
        z_update = map(self.sensor, x_update)
        #calculate the weights for the particles 
        p_w = [scipy.stats.norm(x, np.sqrt( self.varence) ).pdf(z) for x in z_update]
        w = [float(x)/float(np.sum(p_w)) for x in p_w]
        #self.pred_x = self.resample(x_update,w)
        #resample particles
        self.pred_x = self.low_variance_resample(x_update,w)
        #update the time
        self.time = self.time + self.dt
        #return estimated position 
        return np.mean(self.pred_x) 
    
    #basic resmapler 
    def resample(self,x,w):
        sum = cumsum(w)
        return [ x [ np.nonzero(random.random() <= sum )[0][0]] for _ in xrange(self.M) ]
        
        #return [ x [ (sum >= random.random()).argmax() ] for _ in xrange(self.M) ]
    
    #this is an improved resampling method
    def low_variance_resample(self, x, w):
        x_list = []
        inv = 1/float(self.M)
        r = random.uniform(0, inv)
        c = w[0]
        ii = 0
        u = 0
        for m in xrange(self.M):
            u = r + (m-1)*inv
            while u > c :
                ii = ii + 1
                c = c + w[ii]
            x_list.append(x[ii])
        return x_list
                
        
        
    
        
        
        
        