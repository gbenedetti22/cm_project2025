[X, y] = training_data("abalone");
X = zscore(X);

sigma = 0.5;

% Fissi i migliori iperparametri precedenti
best_lr = 1e-7;
best_momentum = 0.3;
best_scale = 1e-5;

% Range di ricerca per theta ed epsilon
theta_values = 0.1:0.1:0.9;                     % [0.1, 0.2, ..., 0.9]
epsilon_values = 0.001:0.001:0.01;              % [0.001, 0.002, ..., 0.01]
epsilon_values = [epsilon_values, 0.02:0.01:0.1]; % unione con [0.02, ..., 0.1]

param_grid = combvec(theta_values, epsilon_values)';

best_mse = inf;
best_params = struct();

for i = 1:size(param_grid, 1)
    theta = param_grid(i, 1);
    epsilon = param_grid(i, 2);

    lbm_params = struct(...
        'tol',             1e-2, ...
        'theta',           theta, ...
        'lr',              best_lr, ...
        'momentum',        best_momentum, ...
        'scale_factor',    best_scale, ...
        'max_constraints', 60 ...
    );

    lbm = LBM(lbm_params);
    svr_params = struct(...
        'max_iter',        90, ...
        'kernel_function', RBFKernel(sigma), ...
        'C',               1, ...
        'epsilon',         epsilon, ...
        'opt',             lbm ...
    );

    svr_lbm = SVR(svr_params);

    try
        [x_lbm, f_values_lbm] = svr_lbm.fit(X, y);
        y_pred_lbm = svr_lbm.predict(X);
        mse_val = mse(y_pred_lbm, y);
    catch ME
        warning("Errore con [theta=%.2f, epsilon=%.3f]: %s", ...
            theta, epsilon, ME.message);
        mse_val = inf;
    end

    if mse_val < best_mse
        best_mse = mse_val;
        best_params.theta = theta;
        best_params.epsilon = epsilon;
    end

    % Log per ogni combinazione
    fprintf("Test %d/%d - theta=%.2f, epsilon=%.3f -> MSE=%.5f | Best MSE=%.5f\n", ...
        i, size(param_grid, 1), theta, epsilon, mse_val, best_mse);
end

fprintf("\n=== Migliori iperparametri trovati ===\n");
fprintf(" - theta:    %.2f\n", best_params.theta);
fprintf(" - epsilon:  %.3f\n", best_params.epsilon);
fprintf("MSE corrispondente: %.5f\n", best_mse);
