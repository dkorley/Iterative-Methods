# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:19:45 2024

@author: Kali Linux
"""

import numpy as np

def JOR(A, b, w, K):
    #get the dimensions of A
    m, n = A.shape
    
    #check if matrix is square matrix
    if m != n:
        raise ValueError("Matrix must be square")
    
    #initialize the matrices
    E = np.zeros((m, n))
    F = np.zeros((m, n))
    
    
    #initialize the iteration x[0]
    x = 100*np.ones((n, 1))
    
    #actual solution
    actual = np.array([[1, 0, -1, 0, 0, -3, 3, 0, 2, -5]]).T
    
    #split A into A = P - N, where P = D = diagonal
    D = np.diag(np.diag(A))    #diagonal matrix
    N = D - A    
     
    for i in range(m):
        for j in range(n):
            #split P = E + F, where E is upper triangular and F is lower triangular matrix
            if i < j:
               F[i, j] = N[i, j]    #upper triangular
            else:
               E[i, j] = N[i, j]    #lower triangular
                  
    #compute the jacobi iteration for x_k+1 = MX_k + Nb  
    for k in range(1, K):
        x_new = (np.eye(m, n) - np.dot(w, (np.linalg.inv(D) @ A)))@x + np.dot(w, np.linalg.inv(D)@b)
       
        #check for convergence
        e_k = np.linalg.norm((actual - x_new), 2)
        if e_k < 1e-8:
            return k+1   # Return the iteration index at which convergence occurred
        
        x = x_new
               
    return K   # If convergence is not reached within K iterations, return K 



















            