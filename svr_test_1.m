clc; clear; close all;

% Migliori iperparametri abalone:
%   θ = 0.6
%   σ = 0.4
%   ε = 0.10
% MSE: 4.1976 time: 334.2511
% MSE (LBM): 4.1966 time: 148.3282

% Migliori iperparametri airfoil:
%   θ = 0.4
%   σ = 0.7
%   ε = 0.01
%   C=3
%gap lbm-oracle 2.2644e-06
% relative gap lb-ub 7.756187e-06
%tempo oracle 128.8039
%tempo lbm 387.4924
% MSE: 6.5557
% MSE (LBM): 6.5524

%%-----%%

% Migliori iperparametri wine:
%   θ = 0.4
%   σ = 0.7
%   ε = 0.01
%   C=3
%gap lbm-oracle 2.2339e-05
% relative gap lb-ub 2.000e-07
%tempo oracle 74.6844
%tempo lbm 297.9364
% MSE: 0.0069757
% MSE (LBM): 0.0075217


% Migliori iperparametri white wine:
%   θ = 0.8
%   σ = 0.5
%   ε = 0.01
% MSE: 0.064935 time: 438.608
% MSE (LBM): 0.064731 time: 268.0662
[X, y] = training_data("red_wine");
X = zscore(X);

sigma = 0.7;
lbm_params = struct(...
        'tol',             1e-8, ...
        'theta',           0.6, ...
        'max_constraints', 100 ...
    );

    lbm = LBM(lbm_params);
    svr_params = struct(...
        'max_iter',        150, ...
        'kernel_function', RBFKernel(sigma), ...
        'C',               3, ...
        'epsilon',         0.01, ...
        'opt',             lbm ...
    );

svr = SVR(rmfield(svr_params, 'opt'));
svr_lbm = SVR(svr_params);

[x_lbm, f_values_lbm, f_times_lbm] = svr_lbm.fit(X, y);
[x, f_values, f_times] = svr.fit(X, y);

y_pred = svr.predict(X);
y_pred_lbm = svr_lbm.predict(X);

disp("MSE: " + mse(y_pred, y));
disp("MSE (LBM): " + mse(y_pred_lbm, y));


plot_gap(x, f_values, x_lbm, f_values_lbm);
plot_time(f_values, f_times, f_values_lbm, f_times_lbm);
disp ("time lbm: "+ f_times_lbm)
disp ("time oracle: "+ f_times)

% figure; hold on;
% plot(X, y, '-', 'LineWidth', 1, 'DisplayName', 'Training data');
% plot(X, y_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'SVR Predictions');
% xlabel('X'); ylabel('y'); title('SVR (with RBF Kernel) using LBM', 'FontSize', 22);
% legend('FontSize', 18); grid on;