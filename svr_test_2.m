clc; clear; close all;

[X, y] = training_data("abalone");
X = zscore(X);

sigma = 0.5;
lbm_params = struct(...
    'tol',             1e-6, ...
    'theta',           0.9, ...
    'max_constraints', 100 ...
    );

lbm = LBM(lbm_params);
svr_params = struct(...
    'max_iter',        300, ...
    'kernel_function', RBFKernel(sigma), ...
    'C',               1, ...
    'epsilon',         1e-6, ...
    'opt',             lbm ...
    );

svr = SVR(rmfield(svr_params, 'opt'));
svr_lbm = SVR(svr_params);

[x, h] = svr.fit(X, y);
[x_lbm, h_lbm] = svr_lbm.fit(X, y);

f_values = h.f_values;
f_times = h.f_times;
f_values_lbm = h_lbm.f_values;
f_times_lbm = h_lbm.f_times;

y_pred = svr.predict(X);
y_pred_lbm = svr_lbm.predict(X);

disp(min(f_values_lbm));

gap = abs(min(f_values_lbm) - min(f_values)) / (abs(min(f_values)));

disp("MSE: " + mse(y_pred, y));
disp("MSE (LBM): " + mse(y_pred_lbm, y));
disp("Gap: " + gap);

plot_gap(f_values, f_values_lbm);
plot_time(f_values, f_times, f_values_lbm, f_times_lbm);