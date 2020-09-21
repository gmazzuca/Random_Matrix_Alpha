# Simulation of the mean density of states for the Laguerre alpha ensemble, for a general definition of this matrix ensemble look at [1]

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import os
##### MAIN ####

n = 1000 # size of the matrix
trials = 1000 # number of trials
gamma_values = [0.8,0.4] # values of gamma 
alpha_values = [1,10,50,100] # values of alpha
for gamma in gamma_values: # Limite N/beta
    for alpha in alpha_values: # Limite N/M
        eig = []
        for k in range(trials): # esperimento vero e proprio
            B = np.zeros((n, int(n/gamma)))
            B[0,0] = np.sqrt(np.random.chisquare(2*alpha/gamma,1))
            for kk in range(n-1): # Matrice B
                B[kk+1,kk]  = np.sqrt(np.random.chisquare(2*alpha,1))
                B[kk+1,kk+1] = np.sqrt(np.random.chisquare(2*alpha/gamma,1))
            J = np.dot(B, B.transpose())*gamma*0.5/alpha # Matrice J
            eig = np.append(eig, LA.eigvalsh(J))
        plt.hist(eig,bins= 300, density = 1, alpha = 0.7, color = 'r')
        plt.title(r'$\alpha = %.0f, \, \gamma = %.3f$' %(alpha,gamma))
        plt.show()
        
