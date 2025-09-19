import numpy as np
from sklearn import svm
import random
import math

import math

NBVECS=10000
VECDIM=2

ballradius=0.5
x=ballradius*np.random.randn(NBVECS,VECDIM)

xa= np.zeros((NBVECS,2))

angle = 2.0 * math.pi * np.random.randn(1, NBVECS)
radius = 3.0 + 0.1 * np.random.randn(1, NBVECS)
xa = np.zeros((NBVECS,2))
xa[:, 0] = radius * np.cos(angle)
xa[:, 1] = radius * np.sin(angle)
X_train = np.concatenate((x, xa))
Y_train = np.concatenate((np.zeros(NBVECS), np.ones(NBVECS)))



clf = svm.SVC(kernel='poly', gamma='auto', coef0=1.1)
clf.fit(X_train, Y_train)

test1 = np.array([0.4,0.1])
test1 = test1.reshape(1,-1)
predicted1 = clf.predict(test1)
print(predicted1)



supportShape = clf.support_vectors_.shape
nbSupportVectors = supportShape[0]
vectorDimensions = supportShape[1]
print("nbSupportVectors = %d" % nbSupportVectors)
print("vectorDimensions = %d" % vectorDimensions)
print("degree = %d" % clf.degree)
print("coef0 = %f" % clf.coef0)
print("gamma = %f" % clf._gamma)
print("intercept = %f" % clf.intercept_)


dualCoefs = clf.dual_coef_
dualCoefs = dualCoefs.reshape(nbSupportVectors)
supportVectors = clf.support_vectors_
supportVectors = supportVectors.reshape(nbSupportVectors * VECDIM)
print("Dual Coefs")
print(dualCoefs)
print("Support Vectors")
print(supportVectors)

# prompt: plot the data on graph to visualize the two clustors

import matplotlib.pyplot as plt

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(x[:, 0], x[:, 1], label='Class 0', alpha=0.7)
plt.scatter(xa[:, 0], xa[:, 1], label='Class 1', alpha=0.7)

# Plot the support vectors
#plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           # s=100, linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Data Visualization')
plt.legend()
plt.grid(True)
plt.show()

angle = 2.0 * math.pi * np.random.randn(1, NBVECS)

print(Y_train)



