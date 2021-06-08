# Mean density of states for the Antisymmetric Gaussian Beta ensemble in the high temperature regime
# For a definition of this matrix ensemble see [1]

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import matplotlib.pylab as pylab
import mpmath as mp
from mpmath import *
from scipy.special import gamma
from scipy.special import hyperu
from scipy.stats import gamma as GammaDistribution

params = {'legend.fontsize': 30,
          'figure.figsize': (20, 13),
         'axes.labelsize': 45,
         'axes.titlesize':50,
         'xtick.labelsize':30,
         'ytick.labelsize':30}
pylab.rcParams.update(params)


def rho(alpha, x):
    # Theoretical density
    mp.dps = 25; mp.pretty = True
    # connection formula for whittaker function
    mywhit = lambda k,z : 1j*whitw(-k,0,-z)*gamma(0.5+k)/gamma(0.5-k) + gamma(0.5+k)*whitm(-k,0,-z)*np.exp(k*np.pi*1j)/gamma(1)
    return np.real(abs(x)/(gamma(alpha)*gamma(alpha+1)*abs(mywhit(-alpha+0.5,-x*x))**2))




particles = 2000 # number of particles
trials = 100 # number of trials
alpha_values = [2.5,1.5,0.5]

for j,alpha in enumerate(alpha_values): # Limit N*beta
    eig = []
    # Simulation of eigenvalues
    for k in range(trials):
            q = np.zeros(particles-1)
            for l in range(particles-1):
                q[l] = np.sqrt(np.random.gamma(alpha*(1- (l+1)/particles),scale = 1,size = 1 ))
            B = -np.diag(q,-1) + np.diag(q,1)
            eig = np.append(LA.eigvals(B), eig)# computation of the eigenvalues

    #Save and plots
    fname = 'Gaussian_antisym_alpha_%0.3f' %alpha
    np.savetxt('%s.dat' %fname, eig)
    limit = 4
    plt.hist(np.imag(eig),bins= 300,range = (-limit,limit), density = True, alpha = 0.7, color = 'r',label = 'Numeric')
    xfine = np.linspace(-limit,limit,num=500)
    mp.dps = 25; mp.pretty = True
    rhoval = np.zeros(len(xfine))
    for l,x in enumerate(xfine):
        rhoval[l] = rho(alpha,x)
    
    plt.plot(xfine, rhoval, 'k-',linewidth = 2, label = 'Theoretical')
    plt.title(r'$\alpha = %0.3f$' %(alpha))
    plt.legend(loc = 1)
    plt.savefig('%s.png' %(fname))
    plt.close()

#G. Mazzuca, and P.J. Forrester: The classical beta ensembles with  beta proportional to 1/N: from loop equations to Dyson's disordered chain. arXiv e-print 2102.09201 (2021)
