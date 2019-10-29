#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 00:21:29 2019

@author: faisal
"""
# This is an example illustrates the Conjugate gradient optimization method
import numpy as np
import matplotlib.pyplot as plt
# Assume that A in the shape [[2a, b], [b , 2c]]
# for a quadrtic form ax^2 + b xy + b y^2
def Conjugate_gradient(A, b, plot_solution=False):
   Xs = []
   # initial guess
   x = np.random.random(2) * [5,9]
   Xs.append(np.copy(x))
   # compute the initial resdual and choose it as the first direction to optimize
   # accroding to
   r = b - np.dot(A, x)
   d = r
   k = 0
   # optimization loop
   while True:
      alpha = np.dot(d,r) / np.dot(d, np.dot(A,d))
      x += (alpha * d)
      r_new = r - alpha * np.dot(A,d)
      print ('At k = {}, x={}'.format(k,x))
      if np.sqrt(np.dot(r,r)) < 1e-3:
         break
      Xs.append(np.copy(x))
      beta = np.dot(r_new,r_new) / np.dot(r, r)
      d = r_new + beta * d
      k += 1
      r  = r_new
   print(k)
   print(np.dot(A, x))
   if plot_solution:
      plot_func_and_gradient(A, b, Xs)
   return x
# Visulize the function and its solution
def plot_func_and_gradient( A, b, Xs,x_range=[-6,6], y_range=[-6,6], grid_num_points=100, num_countors=30):
   X = np.linspace(x_range[0], x_range[1], grid_num_points)
   Y = np.linspace(y_range[0], y_range[1], grid_num_points)
   for i in range(len(Xs) - 1):
      plt.plot([Xs[i][0],Xs[i+1][0]],[Xs[i][1],Xs[i+1][1]] )
   X, Y= np.meshgrid(X, Y)
   Z = 0.5 * A[0][0] * X**2 + 0.5 * A[1,1] * Y**2 + A[0][1] * X * Y
   plt.contour(X, Y, Z,num_countors)
   plt.show()
   
if __name__ == '__main__':
   A = np.array([[2/3, 1/5], [1/5, 2/3]])
   b = np.array([0, 0])
   Conjugate_gradient(A, b, True)