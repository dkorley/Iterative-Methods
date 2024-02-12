function [x, e_k] = jacobi_iteration(A, b, K)
    %dimensions of A
    [m, n] = size(A);


    %initialize the matrices
    D = zeros(m, n);
    N = zeros(m, n);
    E = zeros(m, n);
    F = zeros(m, n);
    
    %split A into D-diagonal, F-upper triangular and E-lower triangular
    for i = 1: m
        D(i, i) = A(i, i);   %diagonal matrix 
        for j = 1: n
            N(i, j) = D(i, j) - A(i, j);

            if i < j
                F(i, j) = N(i, j);    %upper triangular matrix
            else
                E(i, j) = N(i, j);    %lower triangular matrix
            end
        end
    end

    %initial vector
    x = [zeros(n, 1)];

    %actual solution
    %actual = [1; -1; 0; 2];
    actual = [1; 0; -1; 0; 0; -3; 3; 0; 2; -5];

    %compute the iterative scheme
    for k = 1: K
        x = (eye(m, n)-(inv(D) * A))*x + inv(D)*b;
    end
   
    e_k = norm((actual - x), 2);
    
    %check if the jacobi method converges
    eigenvalue = eig(eye(m, n) - (inv(D) * A));

    if max(abs(eigenvalue)) < 1
        disp("The Jacobi method converges");
    else
        disp("The jacobi method does not converge")
    end
    
    disp(' ')
    
    % Display the last computed iterate and the iteration number
    disp(['The ' num2str(K) 'th iteration computed:']);
    disp(round(x,2));

    % Display the K-th error vector
    disp(['The ' num2str(K) 'th error vector ||e_' num2str(K) '||:']);
    disp(round(e_k, 2));
    
end