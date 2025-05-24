function [svr_params_lbm, svr_params_oracle] = get_params(dataset)
    switch lower(dataset)
        
        case "abalone"
            sigma = 0.4;
            epsilon = 1e-6;
            lbm_params = struct('tol', 2e-5, 'theta', 0.6, 'max_constraints', 100);
        
        case "white_wine"
            sigma = 0.6;
            epsilon = 1e-6;
            lbm_params = struct('tol', 2e-5, 'theta', 0.7, 'max_constraints', 100);
        
        case "red_wine"
            sigma = 0.55;
            epsilon = 1e-6;
            lbm_params = struct('tol', 1e-6, 'theta', 0.7, 'max_constraints', 100);
         
        case "airfoil"
            sigma = 0.7;
            epsilon = 1e-7;
            lbm_params = struct('tol', 1e-6, 'theta', 0.6, 'max_constraints', 100);
        
        otherwise
            error("No parameters defined for: " + dataset);
    end

    svr_params_oracle = struct( ...
        'max_iter',        300, ...
        'kernel_function', RBFKernel(sigma), ...
        'C',               1, ...
        'epsilon',         epsilon ...
    );

    svr_params_lbm = struct( ...
        'max_iter',        300, ...
        'kernel_function', RBFKernel(sigma), ...
        'C',               1, ...
        'epsilon',         epsilon, ...
        'opt',             LBM(lbm_params) ...
    );
    
end
