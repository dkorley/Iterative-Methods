# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 14:29:11 2024

@author: Kali Linux
"""

import numpy as np

def guass_seidel(A, b, K):
    #get the dimensions of A
    m, n = A.shape
    
    #check if matrix is square matrix
    if m != n:
        raise ValueError("Matrix must be square")
    
    #initialize the matrices
    D = np.zeros((m, n))
    N = np.zeros((m, n))
    E = np.zeros((m, n))
    F = np.zeros((m, n))
    
    
    #initialize the iteration x[0]
    x = [np.zeros((n, 1))]
    
    #actual solution
    actual = np.linalg.solve(A, b)
    
    #split A into A = P - N, where P = D = diagonal
    for i in range(m):
        D[i, i] = A[i, i]
    
        # P = D - A
        for j in range(n):
            N[i, j] = D[i, j] - A[i, j]
            
            #split P = E + F, where E is upper triangular and F is lower triangular matrix
            if i < j:
               F[i, j] = N[i, j]    #upper triangular
            else:
               E[i, j] = N[i, j]    #lower triangular
    
    #compute the guass-seidel iteration
    k = 1
    while k <= K:
        x_new = (np.linalg.inv(D - E)@F) @ x + np.linalg.inv(D - E)@b
        k += 1
        
        error = np.linalg.norm(actual - x_new)
        if error < 1e-8:
            return x_new, k+1
        x = x_new   
            
    return x_new, K
        
A = np.array([[4, -1, 2, 0], [-1, 4, -1, 1], [2, -1, 6, -2], [0, 1, -2, 4]])
b = np.array([[5, -3, -1, 7]]).T
K = 200

result, K = guass_seidel(A, b, K)

print(f"\nThe {K}{''.join(['st', 'nd', 'rd', 'th'][min(K-1, 3)])} computed iterate:")
print(result)
print(f"The method converged at the K = {K} iterate")