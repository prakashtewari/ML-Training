# -*- coding: utf-8 -*-
"""
Created on Thu May 10 19:30:48 2018

@author: Prakash.Tiwari
"""

#Gradient descent
import pandas as pd
import pylab
import numpy as np
# from sklearn.datasets.samples_generator import make_regression 
import pylab
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import os
import ipdb

os.chdir(u'.\data')

###########################
###### Define Parameters
###########################
 
alpha = 0.01 # learning rate
ep = 0.01 # convergence criteria

###########################
###### Read file
###########################
df = pd.read_csv('data.csv')
x = df.iloc[:, [1,2]]
y = df.iloc[:, [3]]
# print ('df.shape = ' + df.shape)

###########################
###### Cost function
###########################
def compute_cost_function(theta, x, y):
    hypo = np.dot(x,theta)
    return 1/2.0/m*np.sum((hypo - y)**2)
    
###########################
###### Gradient Descent function
###########################
def gradient_descent(alpha, x, y, ep=0.0001, max_iter=1500):
    converged = False
    iter = 0
    m = x.shape[0] # number of samples

    try:
        n = x.shape[1]
    except:
        n = 1        
        
    # initial theta
    theta = np.random.rand(n+1)
    
    x = x.as_matrix()        
    x_ones = np.ones(m)
    x = np.c_[x_ones, x]
    
    y = y.as_matrix()
        
    # total error, J(theta)
    J = compute_cost_function(theta, x, y)
    print('J= %.2f' %J)
    
    # Iterate Loop
    num_iter = 0
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        hypo = np.dot(x,theta)
        grad = (np.dot((hypo - y), x))*1/m

        # update the theta_temp
        theta = theta - alpha*grad

        # mean squared error
        e = compute_cost_function(theta, x, y)
        print ('J = %.2f'%e)
        J = e   # update error 
        num_iter += 1  # update iter
    
        if num_iter == max_iter:
            print ('Max interactions exceeded!')
            converged = True

    return theta

# call gredient decent, and get intercept(=theta0) and slope(=theta1)
theta_final = gradient_descent(alpha, x, y, ep, max_iter=1500)

#linear regression from stats
import stats
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)

