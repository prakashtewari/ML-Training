"""

Implementation of SVM in Python using dummy data - Social_Network_Ads.csv

@author: Prakash.Tiwari
"""
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat  
from sklearn import svm
import os
os.chdir(u'.\data')

###########################
###### Read Data
###########################

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, [2, 3]]
y = df.iloc[:, 4]

###########################
####### Feature Scaling
###########################
sc = StandardScaler()
X = pd.DataFrame(sc.fit_transform(X))

###########################
####### Visualise the data - Only for 2D
###########################
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')

###########################
####### Modeling
###########################

"""
Linear SVM:
    a.    LIBSVM - Use SVC from svm class (with kernel = 'linear')
    b.    Liblinear - Use LinearSVC from svm class -it has more flexibility in the choice of
        penalties and loss functions and should scale better to large numbers of
        samples.
"""

# Create SVM classification object 
model = svm.SVC(kernel = 'linear', C = 1, gamma = 1)
#svc = svm.LinearSVC(C = 1)
model.fit(X,y)
                     
#Print Model accuracy and confusion Matrix
print_accuracy(model, X, y)

#Plot Decision functions - Only for 2D SVM
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)              


"""
Non-Linear SVM:
    a.'rbf' for Gaussian Kernel - Increasing C will lead to overfitting and generalizaton error
    b.'poly' for polynomial Kernel
"""
    
# Create SVM classification object 
model = svm.SVC(kernel = 'rbf', C = 1, gamma = 1)
model.fit(X,y)

#Print Model accuracy and confusion Matrix
print_accuracy(model, X, y)

#Plot Decision functions  - Only for 2D SVM
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)                  

##################
## Model Accuracy - Score, Confusion Matrix
##################          
def print_accuracy(model, X, y):
    """Print Model accuracy and confusion matrix"""   
    predicted= model.predict(X)
    #model score - correct prediction/ (sample size)
    score = model.score(X,y)
    
    print 'Score {}'.format(score*100)
    print 'Confusion Matrix: \n'+ str(confusion_matrix(y, predicted))            
            
            
##################
## Plot Decision functions - For a 2 variable SVM only
##################            
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], 
               alpha=0.5, 
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)    

 

