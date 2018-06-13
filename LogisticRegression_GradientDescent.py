# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:43:15 2018

@author: Prakash.Tiwari
"""
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import os
os.chdir(u'.\data')


###########################
###### Define Parameters
###########################
 
alpha = 0.5 # learning rate
ep = 0.01 # convergence criteria

df = pd.read_csv('Data.csv')
x = df.iloc[:, [2, 3]]
y = df.iloc[:, 4]

###########################
####### Feature Scaling
###########################
sc = StandardScaler()
x = pd.DataFrame(sc.fit_transform(x))

###########################
####### Sigmoid function
###########################
def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

#x.to_clipboard()
###########################
###### Cost function
'''
Hypothesis function (hypo()) = sigmoid(theta.X)

1. Y = 1 --> Cost(hypo, y) = - log(hypo())
2. Y = 0 --> Cost(hypo, y) = - log(1- hypo())

if Cost = 0 then hypo() = y
if Cost --> infinity and y = 0, then hypo() = 1
if Cost --> infinity and y = 1, then hypo() = 0

So --> 
Cost(hypo, y) =1/2m* [ y * (log(hypo())) - (1-y)*(log(hypo()))]

'''
###########################
def compute_cost_function(theta,m, x, y):
    z = np.dot(x,theta)
    hypo = np.asarray(sigmoid(z))
    return 1/float(m)*(np.matmul(-y.T,np.log(hypo)) - np.dot((1-y).T, np.log(1- hypo)))
    
    
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
    theta = np.ones(n+1)
    
    x = x.as_matrix()        
    x = np.c_[np.ones(m), x]
    
    y = y.as_matrix()
        
    # total error, J(theta)
    J = compute_cost_function(theta,m, x, y)
    print('J= %.2f' %J)
    
    # Iterate Loop
    num_iter = 0
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        hypo = sigmoid(np.dot(x,theta))
        grad =(np.dot((hypo - y), x))/m

        # update the theta_temp
        theta = theta - alpha*grad

        # mean squared error
        e = compute_cost_function(theta,m, x, y)
        print ('J = {}'.format(e))
        J = e   # update error 
        num_iter += 1  # update iter
    
        if num_iter == max_iter:
            print ('Max interactions exceeded!')
            converged = True

    return theta


# call gredient decent, and get intercept(=theta0) and slope(=theta1)
theta_final = gradient_descent(alpha, x, y, ep, max_iter=1000)
print('Gradient Descent Parameters: {}'.format(theta_final))


###########################
###### SKlearn Logistic regression
###########################
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(fit_intercept = True, random_state = 0, C = 1e15)
classifier.fit(x, y)

#Coefficients
print('Intercept: %.2f ' %(classifier.intercept_))
print('Coefficients: {}'.format(classifier.coef_))
