function iteration = gradient_method(A, b, x0, tol, K)
    x = x0;  %initialize the vector
    x_exact = inv(A) * b;  %exact solution
    r = b - (A * x);  %compute residual

    for k = 1:K
        alpha = (r'*r) ./ (r'*A*r); %compute step size
        x_new = x + (alpha * r);  %comptute new approximation
        r = b - A * x_new;  %update residual

        %check for convergence
        if norm(x_exact - x_new) < tol
            iteration = k;  %return the iteration index at which convergence occurred
            return;
        end
        x = x_new;
    end
    iteration = K;  %return maximum iteration
end