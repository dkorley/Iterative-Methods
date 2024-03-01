function iter_counts = richardson_method(A, b, x0, alpha, tol, max_iter)

    x = x0;  % Initial guess for the solution
    x_exact = inv(A)*b; %exact solution
  
    for k = 1:max_iter
        residual = b - A*x;   %compute the residual vector
        x_new = x + alpha * residual;  %compute the Richardson iterationn

        %check for convergence
        if norm(x_exact - x_new) < tol
            iter_counts = k;    %return the iteration index at which convergence occurred
            return;
        end

        x = x_new;
    end

    iter_counts = max_iter;   % return the number of iterations
end


