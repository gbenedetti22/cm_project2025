clc; clear; close all;

[X, y] = training_data("abalone");
X = zscore(X);

theta_values = 0.1:0.1:0.9;
sigma_values = 0.1:0.1:0.9;

% param_grid = combvec(lr_values, momentum_values, scale_values, C_values)';
param_grid = combvec(theta_values, sigma_values)';

best_mse = inf;
best_params = struct();

for i = 1:size(param_grid, 1)
    theta = param_grid(i, 1);
    sigma = param_grid(i, 2);

    lbm_params = struct(...
        'tol',             1e-2, ...
        'theta',           theta, ...
        'max_constraints', 60 ...
    );

    lbm = LBM(lbm_params);
    svr_params = struct(...
        'max_iter',        60, ...
        'kernel_function', RBFKernel(sigma), ...
        'C',               1, ...
        'epsilon',         0.05, ...
        'opt',             lbm ...
    );

    svr_lbm = SVR(svr_params);

    try
        [x_lbm, f_values_lbm] = svr_lbm.fit(X, y);
        y_pred_lbm = svr_lbm.predict(X);
        mse_val = mse(y_pred_lbm, y);
    catch ME
        warning("Errore con combinazione [%g, %g, %g]: %s", lr, momentum, scale_factor, ME.message);
        mse_val = inf;
    end

    if mse_val < best_mse
        best_mse = mse_val;

        best_params.sigma = sigma;
        best_params.theta = theta;
    end

    fprintf("Test %d/%d | Best MSE=%.5f\n", ...
        i, size(param_grid, 1), best_mse);
end

fprintf("Migliori iperparametri trovati:\n");
fprintf(" - theta: %g\n", best_params.theta);
fprintf(" - sigma: %g\n", best_params.sigma);

fprintf("MSE corrispondente: %.5f\n", best_mse);
