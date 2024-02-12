function iterations = JOR(A, b, w, K)
    %dimensions of A
    [m, n] = size(A);

    %initialize the matrices
    E = zeros(m, n);
    F = zeros(m, n);

   %split A into D-diagonal, F-upper triangular and E-lower triangular
    D = diag(diag(A));   %diagonal matrix 
    N = D - A;
    for i = 1: m  
        for j = 1: n
            if i < j
                F(i, j) = N(i, j);    %upper triangular matrix
            else
                E(i, j) = N(i, j);    %lower triangular matrix
            end
        end
    end

    %initial vector
    x = 100*ones(n, 1);

    %actual solution
    actual = [1; 0; -1; 0; 0; -3; 3; 0; 2; -5];

    % Compute the Jacobi iteration for x_{k+1} = M*x_k + N*b
    for k = 1: K
        x_new = w*inv(D) * (1/w *D - A) * x + w * inv(D) * b;
        
        %check for convergence
        e_k = norm((actual - x_new), 2);
        if e_k < 1e-8
            iterations = k+1;    %return the iteration index at which convergence occurred
            return;
        end
        x = x_new;
    end

    iterations = K;  % return the number of iterations
end
