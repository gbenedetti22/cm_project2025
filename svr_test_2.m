clc; clear; close all;

% Caricamento e preprocessing dati
[X, y] = training_data("abalone");
X = zscore(X); 

% Parametri comuni
C = 1;
epsilon = 0.05;
tol = 1e-5;


max_constraints = 100;
lambda=0.6;
sigma = 0.9;
lbm_params = struct(...
    'max_iter', 100, ...       
    'epsilon', 1e-4, ...
    'tol', 1e-6, ...
    'theta', 0.05, ...
    'max_constraints', max_constraints, ...
    'lambda', lambda);

lbm = LBM(lbm_params); % Passa i parametri corretti

% Configurazione SVR
kernel = RBFKernel(sigma);

svr_params_fmincon = struct(...
    'kernel_function', kernel, ...
    'C', C, ...
    'epsilon', epsilon, ...
    'tol', tol);

svr_params_lbm = struct(...
    'kernel_function', kernel, ...
    'C', C, ...
    'epsilon', epsilon, ...
    'opt', lbm, ...
    'tol', tol);

% Addestramento modelli
svr_fmincon = SVR(svr_params_fmincon);
svr_lbm = SVR(svr_params_lbm);

tic;
[alpha_fmincon, f_hist_fmincon] = svr_fmincon.fit(X, y);
time_fmincon = toc;

tic;
[alpha_lbm, f_hist_lbm] = svr_lbm.fit(X, y);
time_lbm = toc;

% Confronto con MATLAB
tic;
svr_matlab = fitrsvm(X, y, ...
    'KernelFunction', 'rbf', ...
    'KernelScale', sigma, ...
    'BoxConstraint', C, ...
    'Epsilon', epsilon, ...
    'Standardize', false);
time_matlab = toc;

% Predizioni e MSE
y_pred_fmincon = svr_fmincon.predict(X);
y_pred_lbm = svr_lbm.predict(X);
y_pred_matlab = predict(svr_matlab, X);

fprintf('=== Tempi di Esecuzione ===\n');
fprintf('fmincon: %.2f s\n', time_fmincon);
fprintf('LBM:     %.2f s\n', time_lbm);
fprintf('MATLAB:  %.2f s\n\n', time_matlab);

fprintf('=== MSE ===\n');
fprintf('fmincon: %.4f\n', mse(y_pred_fmincon, y));
fprintf('LBM:     %.4f\n', mse(y_pred_lbm, y));
fprintf('MATLAB:  %.4f\n', mse(y_pred_matlab, y));

% Plot convergenza
plot_gap(f_hist_fmincon(end), f_hist_fmincon, ...
         f_hist_lbm(end), f_hist_lbm);