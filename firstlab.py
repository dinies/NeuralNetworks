#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:25:32 2016

@author: edoardoghini
"""

import numpy as np

N= 15

x= np.random.rand(N)*3

y= np.sin(x) + np.random.randn(N)*0.1


def power_exp(x,p=3):
    coeffs= np.arange(0,p+1)
    x= x.reshape(x.shape[0],1)**coeffs.reshape(1,p+1)
    return x

P=1
                 
X = power_exp(x)

A= X.T.dot(X) +0.01*np.eye(P+1)
b = X.T.dot(y)
w = np.linalg.inv(A).dot(b)

import matplotlib.pyplot as plt

plt.figure()

plt.scatter(x,y)

x_sampling= np.arange(0, 3, 0.01)
plt.plot(x_sampling, np.sin(x_sampling), 'r')


x_s_exp = power_exp(x_sampling,P)
plt.plot(x_sampling, x_s_exp.dot(w),'r')

plt.show()