# Simulation of the mean density of states of the jacobi alpha ensemble, for a general definition of this matrix ensemble look at [1]

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from numpy.random import beta
##### MAIN ####

n = 1000 # size of the matrix
trials = 1000 # trials
a_values = [0.8, 1.5, 25.8] # values of the parameter a
b_values = [2,10] # values of the parameter b
alpha_values = [1,10,50,100] # values of the parameter alpha

for a in a_values: # a parameter for Beta
    for b in b_values: # b parameter for Beta
        for alpha in alpha_values: # alpha parameter 
            eig = []
            for k in range(trials):
            
                # random vectors (p,q)
                q = np.append(0,beta(alpha,alpha + a+b+2,n-1))
                p = beta(alpha+a+1,alpha + b+1,n)
            
                # matrix entries
                s = np.sqrt(p*(1-q))
                q = q[1:]
                p = p[:-1]
                t = np.sqrt((1-p)*q)
                
                B = np.diag(s) + np.diag(t,-1)
                J = np.dot(B,B.transpose())
                # eigenvalues computations
                eig = np.append(eig, LA.eigvalsh(J))
            
            plt.hist(eig,bins= 300, density = 1, alpha = 0.7, color = 'r')
            plt.title(r'$\alpha = %.1f, \, a = %.1f,\, b = %.1f$' %(alpha,a,b))
            plt.show()
            
            
        #[1] G. Mazzuca: On the mean Density of States of some matrices related to the beta ensembles and an application to the Toda lattice. arXiv e-print 2008.04604 (2020).
