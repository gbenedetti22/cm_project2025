clc; clear; close all;
% carico e normalizzo i dati
[X, y] = training_data("airfoil");
X = zscore(X);

% vettori discreti di valori consentiti
theta_values = [0.5, 0.6, 0.7, 0.8]; % possibili θ
% sigma_values = [0.8, 0.4, 0.6, 0.7];

f_val= -377.7918;
%%%%&%%%%%%%%%%abs(min(f_values_lbm))-abs(min(f_values))
%%gap_rel = abs(min(f_values_lbm) - min(f_values)) / (abs(min(f_values)) + eps);


% numero di tentativi
N = 4;

% Pre-genera gli indici casuali (senza replacement)
theta_idx   = randi(numel(theta_values),   N, 1);
%sigma_idx   = randi(numel(sigma_values),   N, 1);

% array in cui salvare i risultati
mse_vals    = nan(N,1);
thetas      = nan(N,1);
%sigmas      = nan(N,1);

% parametri fissi
maxIter        = 150;
lbm_tol        = 1e-8;
lbm_max_constr = 60;
C              = 1;
epsilon = 0.01;
sigma = 0.7;
% apri pool (se non è già aperto)
if isempty(gcp('nocreate'))
    parpool;
end

% Parfor loop
parfor i = 1:N
    % estrai i parametri scelti
    theta   = theta_values(  theta_idx(i) );
    % sigma   = sigma_values(  sigma_idx(i) );

    % set LBM
    lbm_params = struct(...
      'tol',             lbm_tol, ...
      'theta',           theta, ...
      'max_constraints', lbm_max_constr ...
    );
    lbm = LBM(lbm_params);

    % set SVR
    svr_params = struct(...
      'max_iter',        maxIter, ...
      'kernel_function', RBFKernel(sigma), ...
      'C',               C, ...
      'epsilon',         epsilon, ...
      'opt',             lbm ...
    );
    svr_lbm = SVR(svr_params);

    % train & valuta
    try
      svr_lbm.fit(X, y);
      y_pred    = svr_lbm.predict(X);
      mse_val   = mse(y_pred, y);
    catch
      mse_val   = Inf;
    end

    % salva su vettori
    mse_vals(i) = mse_val;
    thetas(i)   = theta;
    % sigmas(i)   = sigma;
end

% delete(gcp('nocreate'));

% trova il migliore
[best_mse, idx] = min(mse_vals);
best_params = struct( ...
  'theta',   thetas(idx) ...
);
  % 'sigma',   sigmas(idx) ...
% stampa
fprintf("\nMigliori iperparametri dopo %d trial paralleli:\n", N);
fprintf("  θ = %.1f\n", best_params.theta);
% fprintf("  sigma = %.1f\n", best_params.sigma);
fprintf("MSE = %.5f\n", best_mse);
