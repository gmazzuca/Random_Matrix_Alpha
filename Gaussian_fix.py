import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import os
from scipy.special import gamma, digamma
# Simulation of the mean density of states of the Gaussian Alpha ensemble, for a reference look at [1]

##### MAIN ####

n = 500 # size of the matrix
trials = 1000 # number of trials
a_values = [1,10,50,100] # values of the parameter
bound = 4 # plot xlimit
for a in a_values: #
    eig = []
    for k in range(trials): 
            # creation of the matrix
            p = np.random.normal(0,1,n)
            q = np.sqrt(np.random.chisquare(2*a,n-1)*0.5)
            B = np.diag(p) + np.diag(q,-1) + np.diag(q,1)
            # compute the eigenvalues
            eig = np.append(eig, LA.eigvalsh(B)/np.sqrt(a))
    fname = 'gaussian_eig_n_%d_alpha_%f' %(n,a)
    
    salpha = np.sqrt(a)
    yfine = np.sqrt(a)*np.linspace(-bound,bound,n)
    xfine = np.linspace(-bound,bound,n)
    figure = np.zeros(n)
    plt.hist(eig,bins= 300,range = (-bound,bound), density = 1, alpha = 0.7, color = 'r',label = 'simulation')
    plt.title(r'$\alpha = %.0f$' %(a))
    plt.xlim = (-bound,bound)
    plt.legend(loc = 'upper right')
    plt.show()
