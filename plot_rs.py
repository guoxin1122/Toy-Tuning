#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:42:45 2019

@author: macgx
"""

import optunity
import optunity.metrics

# comment this line if you are running the notebook
import sklearn.svm
import numpy as np
import math
import pandas

#%matplotlib inline
from matplotlib import pylab as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(16)

npos = 200 # 200 positive samples
nneg = 200

delta = 2 * math.pi / npos
radius = 2
circle = np.array(([(radius * math.sin(i * delta),
                     radius * math.cos(i * delta))
                    for i in range(npos)]))

x = list(range(npos))
tuple0 = [radius * math.sin(i * delta) for i in range(npos)]
tuple1 = [radius * math.cos(i * delta) for i in range(npos)]
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(121)
ax1.plot(x, tuple0)
ax2 = fig1.add_subplot(122)
ax2.plot(x, tuple1)

neg = np.random.randn(nneg, 2) # 2 dimensions
pos = np.random.randn(npos, 2) + circle

data = np.vstack((neg, pos))
labels = np.array([False] * nneg + [True] * npos)

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)

ax2.plot(neg[:,0], neg[:,1], 'ro')
ax2.plot(pos[:,0], pos[:,1], 'bo')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
fig2.savefig('dataCloud')

#%%
#######################################

@optunity.cross_validated(x=data, y=labels, num_folds=5, regenerate_folds=True)
def svm_rbf_tuned_auroc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = sklearn.svm.SVC(C=10 ** logC, gamma=10 ** logGamma).fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    auc = optunity.metrics.roc_auc(y_test, decision_values)
    return auc

optimal_rbf_pars, info, _ = optunity.maximize(svm_rbf_tuned_auroc, num_evals=300, logC=[-4, 2], logGamma=[-5, 0])
# when running this outside of IPython we can parallelize via optunity.pmap
# optimal_rbf_pars, _, _ = optunity.maximize(svm_rbf_tuned_auroc, 150, C=[0, 10], gamma=[0, 0.1], pmap=optunity.pmap)

print('**********************************************')
print("Optimal parameters: " + str(optimal_rbf_pars))
print("AUROC of tuned SVM with RBF kernel: %1.3f" % info.optimum)

df = optunity.call_log2dataframe(info.call_log)

################################
cutoff = 0.1
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(xs=df[df.value > cutoff]['logC'],
           ys=df[df.value > cutoff]['logGamma'],
           zs=df[df.value > cutoff]['value'])
ax3.set_xlabel('logC')
ax3.set_ylabel('logGamma')
ax3.set_zlabel('AUROC')
ax3.set_title('Show all values of AUROC.')
fig3.savefig('AUROCcutoff01')

################################
cutoff = 0.95 * info.optimum
fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111, projection='3d')
ax4.scatter(xs=df[df.value > cutoff]['logC'],
           ys=df[df.value > cutoff]['logGamma'],
           zs=df[df.value > cutoff]['value'])
ax4.set_xlabel('logC')
ax4.set_ylabel('logGamma')
ax4.set_zlabel('AUROC')
ax4.set_title('Show AUROC > (0.95 * max)')
fig4.savefig('AUROCcutoff095')

#%%
##################################
minlogc = min(df[df.value > cutoff]['logC'])
maxlogc = max(df[df.value > cutoff]['logC'])
minloggamma = min(df[df.value > cutoff]['logGamma'])
maxloggamma = max(df[df.value > cutoff]['logGamma'])

_, info_new, _ = optunity.maximize(svm_rbf_tuned_auroc, num_evals=2500,
                                   logC=[minlogc, maxlogc],
                                   logGamma=[minloggamma, maxloggamma],
                                   solver_name='grid search')

df_new = optunity.call_log2dataframe(info_new.call_log)

logcs = np.reshape(df_new['logC'].values, (50, 50))
loggammas = np.reshape(df_new['logGamma'].values, (50, 50))
values = np.reshape(df_new['value'].values, (50, 50))

levels = np.arange(0.97, 1.0, 0.001) * info_new.optimum

fig5 = plt.figure(5)
ax5 = fig5.add_subplot(111)

CS = ax5.contour(logcs, loggammas, values, levels=levels, cmap=cm.jet) #CS is ContourSet object
#ax3.clabel(CS, inline=True, fontsize=10)

ax5.set_xlabel('log C')
ax5.set_ylabel('log gamma')
ax5.set_title('Contours of SVM tuning response surface')
cbar = fig5.colorbar(CS)
#cbar.ax3.set_ylabel('values')

fig5.savefig('svm')
fig5.show()




