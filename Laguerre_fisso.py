import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import os
##### MAIN ####
''' matrice fissa tipo Wishart-Laguerre ensemble '''


n = 1000
trials = 1000
gamma_values = [0.8,0.4]
c_values = [1,10,50,100]
for gamma in gamma_values: # Limite N/beta
    for c in c_values: # Limite N/M
        eig = []
        for k in range(trials): # esperimento vero e proprio
            B = np.zeros((n, int(n/gamma)))
            B[0,0] = np.sqrt(np.random.chisquare(2*c/gamma,1))
            for kk in range(n-1): # Matrice B
                B[kk+1,kk]  = np.sqrt(np.random.chisquare(2*c,1))
                B[kk+1,kk+1] = np.sqrt(np.random.chisquare(2*c/gamma,1))
            J = np.dot(B, B.transpose())*gamma*0.5/c # Matrice J
            eig = np.append(eig, LA.eigvalsh(J))
        fname = 'eig_n_%d_gamma_%f_c_%f' %(n,gamma,c)
        np.savetxt("%s.dat" %fname, eig)
        plt.hist(eig,bins= 300, density = 1, alpha = 0.7, color = 'r')
        plt.title(r'$\alpha = %.0f, \, \gamma = %.3f$' %(c,gamma))
        plt.savefig('%s.pdf' %(fname))
        plt.close()
        