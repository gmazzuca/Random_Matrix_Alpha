import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import os
from scipy.special import gamma, digamma
import scipy.integrate as integrate
##### MAIN ####
''' matrice fissa tipo Gaussian ensemble '''


n = 500
trials = 1000
a_values = [1,10,50,100]
bound = 4
for a in a_values: # Limite N/M
    eig = []
    for k in range(trials): # esperimento vero e proprio
        
            p = np.random.normal(0,1,n)
            q = np.sqrt(np.random.chisquare(2*a,n-1)*0.5)
            B = np.diag(p) + np.diag(q,-1) + np.diag(q,1)
            eig = np.append(eig, LA.eigvalsh(B)/np.sqrt(a))
    fname = 'gaussian_eig_n_%d_alpha_%f' %(n,a)
    np.savetxt("%s.dat" %fname, eig)
    
    salpha = np.sqrt(a)
    yfine = np.sqrt(a)*np.linspace(-bound,bound,n)
    xfine = np.linspace(-bound,bound,n)
    figure = np.zeros(n)

#     for y in range(len(yfine)): 
#         denominatore1 = integrate.quad(lambda x : x**(a-1)*np.exp(-x**2*0.5)*np.cos(yfine[y]*x),0,+np.inf)[0]
#         denominatore2 = integrate.quad(lambda x : x**(a-1)*np.exp(-x**2*0.5)*np.sin(yfine[y]*x),0,+np.inf)[0]
#         denominatore = denominatore1*denominatore1 + denominatore2*denominatore2
#         numeratore =integrate.quad(lambda x : np.log(x)*x**(a-1)*np.exp(-x**2*0.5)*np.cos(yfine[y]*x),0,+np.inf)[0]*denominatore1 + denominatore2*integrate.quad(lambda x : np.log(x)*x**(a-1)*np.exp(-x**2*0.5)*np.sin(yfine[y]*x),0,+np.inf)[0] 
#         figure[y] = np.exp(-yfine[y]**2*0.5)/(np.sqrt(2*np.pi))*gamma(a)*(+digamma(a)/denominatore - 2 *numeratore/(denominatore**2))
#     plt.plot(xfine,salpha*figure, 'b', label = 'theoretical')
    plt.xlim = (-bound,bound)
    
    plt.hist(eig,bins= 300,range = (-bound,bound), density = 1, alpha = 0.7, color = 'r',label = 'simulation')
    plt.title(r'$\alpha = %.0f$' %(a))
    plt.xlim = (-bound,bound)
    plt.legend(loc = 'upper right')
    plt.savefig('%s.pdf' %(fname))
    plt.close()
