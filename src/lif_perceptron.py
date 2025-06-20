# -*- coding: utf-8 -*-
"""LIF_perceptron.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MKV-zpp9LfcIOUVfkAuW5u0M7aQUObFI
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

np.random.seed(42)
num_datapoints = 1000
dim = 5
a1 = 0.0
b1 = 1.0
a2 = 1.0
b2 = 2.0
class1 = np.random.uniform(a1,b1,size=(num_datapoints,dim))
class2 = np.random.uniform(a2,b2,size=(num_datapoints,dim))

X = np.concatenate([class1,class2])
X = np.hstack((X, np.ones((X.shape[0], 1))))
print("Shape of original data", X.shape)
Y = np.concatenate([-np.ones(num_datapoints),np.ones(num_datapoints)])

# T = np.max(X[:,:dim],axis=1).reshape(X.shape[0],1)
# T = np.mean(X[:,:dim],axis=1).reshape(X.shape[0],1)
K = 1
T = b2
tau = 10*(b2-a1)
vth = K*(T - (a1+b1+a2+b2)/4)/tau
phi = K*(T-X[:,:dim]) / tau
# phi = K*(1-np.exp(-(T-X[:,:dim])/tau))
phi = np.hstack((phi, np.ones((phi.shape[0], 1))))
print("Shape of transformed data", phi.shape)

svm = SVC(kernel='linear')
svm.fit(X, Y)
if svm.score(X, Y) == 1.0:
    print("Original Data is linearly separable")
else:
    print("Original Data is not linearly separable")
w = svm.coef_[0]  # Weight vector (only for linear kernels)
margin = 1 / np.linalg.norm(w)
print(f"The margin (gamma) for original data is: {margin}")

svm = SVC(kernel='linear')
svm.fit(phi, Y)
if svm.score(phi, Y) == 1.0:
    print("Transformed Data is linearly separable")
else:
    print("Transformed Data is not linearly separable")
w = svm.coef_[0]  # Weight vector (only for linear kernels)
margin = 1 / np.linalg.norm(w)
print(f"The margin (gamma) for transformed data is: {margin}")

def training(X,Y,vth):
  np.random.seed(42)
  w = np.random.uniform(0,1,size=(X.shape[1],1))
  epochs = 1000
  for epoch in  range(epochs):
    error = 0
    for (x,y) in zip(X,Y):
      x = x.reshape(X.shape[1],1)
      y_pred = np.sign(w.T@x-vth)
      if y_pred != y:
        w = w + y*x
        error += 1
    if error == 0:
      print("finished in", epoch, "epochs")
      print("w* is ", w)
      break
  return w

print("Perceptron")
w_per = training(X,Y,0)
print("LIF")
w_lif = training(phi,Y,vth)