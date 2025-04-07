%rng(42);

clc; clear; close all;

[X, y] = training_data("winequality_red");
X = zscore(X);
sigma = 0.5;

lbm_params = struct(...
    'tol',             1e-2, ...
    'theta',           0.4878, ...
    'lr',              4.1169e-06, ...
    'momentum',        0.654, ...
    'scale_factor',    3.7483e-05, ...
    'max_constraints', 60 ....
    );

lbm = LBM(lbm_params);
svr_params = struct(...
    'max_iter',        90, ...
    'kernel_function', RBFKernel(sigma), ...
    'C',               1, ...
    'epsilon',         0.02, ...
    'opt',             lbm ...
    );

svr_params.max_iter = 90;

svr_lbm = SVR(svr_params);
svr = SVR(rmfield(svr_params, 'opt'));

[x, f_values, f_times] = svr.fit(X, y);

[x_lbm, f_values_lbm, f_times_lbm] = svr_lbm.fit(X, y);

y_pred = svr.predict(X);
y_pred_lbm = svr_lbm.predict(X);

disp("MSE: " + mse(y_pred, y));
disp("MSE (LBM): " + mse(y_pred_lbm, y));

% Misura tempo totale per SVR senza LBM
%total_time_svr = f_times(end); % Tempo cumulativo dell'ultima iterazione

% Misura tempo totale per SVR con LBM
%total_time_lbm = f_times_lbm(end); % Tempo cumulativo dell'ultima iterazione

total_time_svr = max(f_times);  % Prendi il massimo valore (equivale all'ultimo iter)
total_time_lbm = max(f_times_lbm);
% Stampa risultati
fprintf('=== Tempi di Esecuzione ===\n');
fprintf('SVR (senza LBM): %.2f secondi\n', total_time_svr);
fprintf('SVR (con LBM):   %.2f secondi\n\n', total_time_lbm);


%plot_gap(x, f_values, x_lbm, f_values_lbm);
plot_time(f_values,f_times, f_values_lbm, f_times_lbm);