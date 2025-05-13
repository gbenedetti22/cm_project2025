clc; clear; close all;

[X, y] = training_data("abalone");
X = zscore(X);

sigma = 0.4;

lbm_params = struct(...
        'tol',             1e-2, ...
        'theta',           0.8, ...
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

[x_lbm, f_values_lbm, f_times_lbm] = svr_lbm.fit(X, y);

y_pred = svr_lbm.predict(X);

disp("MSE (LBM): " + mse(y_pred, y));

% plot_gap(x, f_values, x_lbm, f_values_lbm);
% plot_time(f_values, f_times, f_values_lbm, f_times_lbm);

% figure; hold on;
% plot(X, y, '-', 'LineWidth', 1, 'DisplayName', 'Training data');
% plot(X, y_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'SVR Predictions');
% xlabel('X'); ylabel('y'); title('SVR (with RBF Kernel) using LBM', 'FontSize', 22);
% legend('FontSize', 18); grid on;