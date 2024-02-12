import numpy as np

def SOR(A, b, w, K):
    # Get the dimensions of A
    m, n = A.shape
    
    # Initialize the matrices
    E = np.zeros((m, n))
    F = np.zeros((m, n))
    
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
    
    # Initialize the vector x
    x = 100*np.ones((n, 1))
    
    x_new = (np.eye(m) - w * np.linalg.inv(D) @ E) * x
    M = (1 - w)*np.eye(m) + w * np.linalg.inv(D) @ F
    Nw = w * np.linalg.inv(D) @ b
    
    # # Actual solution
    actual = np.array([[1, 0, -1, 0, 0, -3, 3, 0, 2, -5]]).T
    
    # Compute the iterative scheme
    for k in range(K):
        x_new = M @ x + Nw
        
        # Check for convergence
        e_k = np.linalg.norm((x_new - x), 2)
        if e_k < 1e-8:
            return k + 1  # Return the iteration index at which convergence occurred
    
        x = x_new
    
    # If convergence is not reached within K iterations, return K    
    return K 


