# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:13:46 2024

@author: Kali Linux
"""

import numpy as np
import matplotlib.pyplot as plt
import sRelaxation as sr
import jRelaxation as jr



# Define the range of w values
w_values = np.arange(0.2, 1.3, 0.01)

# Initialize arrays to store the number of iterations for each w
iterations_jor = np.zeros_like(w_values)
iterations_sor = np.zeros_like(w_values)


# Define the matrix A and vector b
A = np.array([[-28, 0, 2, -1, 3, 0, 7, 4, 0, -8],
              [0, 41, 0, 13, 7, 4, -3, 1, -3, -2],
              [-2, 13, -76, 1, 0, 12, 4, 2, 1, 6],
              [1, 3, -4, 37, 9, -7, 2, 8, 0, 0],
              [2, 6, -3, 0, 29, 1, 9, -5, 0, 1],
              [0, -2, 1, 0, 3, 113, -15, 0, 4, 0],
              [23, 0, -7, 3, 0, 10, -218, 46, -2, 3],
              [-17, 18, 1, -2, 1, 0, 0, 60, 9, 11],
              [281, 13, -1, -28, 0, -13, 2, -301, -645, -4],
              [1, 0, 47, 402, -331, 84, -52, 18, 23, 961]])

b = np.array([[31, -17, 22, 32, 24, -377, -673, -55, -943, -5213]]).T

# Define the maximum number of iterations
K = 200

# Perform JOR iterations for each w value
for i, w in enumerate(w_values):
    iterations_jor[i] = jr.JOR(A, b, w, K)

# Perform SOR iterations for each w value
for i, w in enumerate(w_values):
    iterations_sor[i] = sr.SOR(A, b, w, K)
   
# # Plot the results
plt.plot(w_values, iterations_jor, label='JOR',color='#4f8fc6')
plt.plot(w_values, iterations_sor, label='SOR', color='orange')
plt.xlabel('w')
plt.ylabel('Number of iterations')
plt.title('Number of iterations vs. w')
plt.legend()
plt.ylim(0, K)
plt.xlim(0.2, 1.4)
plt.grid(True)
plt.tight_layout()
plt.show()