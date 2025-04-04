clc; clear; close all;

[X, y] = training_data("abalone");
X = zscore(X);

% Parametri SVR
% epsilon = 0.05;
% C = 1;
% maxIter = 90;
% tol = 1e-2;
% theta = 0.9;
% max_constraint = 60;

sigma = 0.5;

lbm_params = struct(...
    'tol',             1e-2, ...
    'theta',           0.5, ...
    'lr',              1e-07, ...
    'momentum',        0.3, ...
    'scale_factor',    1e-05, ...
    'max_constraints', 60 ....
    );

lbm = LBM(lbm_params);
svr_params = struct(...
    'max_iter',        60, ...
    'kernel_function', RBFKernel(sigma), ...
    'C',               1, ...
    'epsilon',         0.05, ...
    'opt',             lbm ...
    );

% svr_params.max_iter = 50;
% svr = SVR(rmfield(svr_params, 'opt'));

% svr_params.max_iter = 50;
svr_lbm = SVR(svr_params);

% [x, f_values, f_times] = svr.fit(X, y);

[x_lbm, f_values_lbm, f_times_lbm] = svr_lbm.fit(X, y);

% y_pred = svr.predict(X);
y_pred_lbm = svr_lbm.predict(X);

% disp("MSE: " + mse(y_pred, y));
disp("MSE (LBM): " + mse(y_pred_lbm, y));


% plot_gap(x, f_values, x_lbm, f_values_lbm);