import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from numpy.random import beta
import shutil
##### MAIN ####
''' matrice fissa tipo Jacobi ensemble '''


n = 1000
trials = 1000
a_values = [0.8, 1.5, 25.8]
b_values = [2,10]
c_values = [1,10,50,100]
for a in a_values: # a parameter for Beta
    for b in b_values: # b parameter for Beta
        for c in c_values: # Limit for Nbeta
            eig = []
            for k in range(trials):
            
                # random variable
                q = np.append(0,beta(c,c+a+b+2,n-1))
                p = beta(c+a+1,c+b+1,n)
            
                #elementi di matrice
                s = np.sqrt(p*(1-q))
                q = q[1:]
                p = p[:-1]
                t = np.sqrt((1-p)*q)
            
                B = np.diag(s) + np.diag(t,-1)
                J = np.dot(B,B.transpose())
                eig = np.append(eig, LA.eigvalsh(J))
            
            fname = 'eig_jacobi_n_%d_a_%f_b_%f_c_%f' %(n,a,b,c)
            np.savetxt("%s.dat" %fname , eig)
            plt.hist(eig,bins= 300, density = 1, alpha = 0.7, color = 'r')
            plt.title(r'$\alpha = %.1f, \, a = %.1f,\, b = %.1f$' %(c,a,b))
            plt.savefig('%s.pdf' %(fname))
            plt.close()
            shutil.move('%s.pdf' %fname, 'Immagini')
        