clc; clear; close all;

% Caricamento e normalizzazione dati
[X, y] = training_data("abalone");
X = zscore(X);

% Definizione griglie di parametri
sigma_values = [0.1, 0.3, 0.5, 0.7, 1.0];
theta_values = [0.1, 0.3, 0.5, 0.7, 0.9];
epsilon_values = [0.001, 0.01, 0.05, 0.1];

% Generazione combinazioni parametri
param_grid = combvec(sigma_values, theta_values, epsilon_values)';
n_combinations = size(param_grid, 1);

% Inizializzazione risultati
gaps = zeros(n_combinations, 1);

parfor i = 1:n_combinations
    disp("Iter: " + i);
    params = param_grid(i, :);
    
    % Estrai parametri
    sigma = params(1);
    theta = params(2);
    epsilon = params(3);
    
    try
        % Costruzione kernel
        kernel = RBFKernel(sigma);

        % Ottimizzatore LBM
        lbm_params = struct(...
            'tol',             1e-6, ...
            'theta',           theta, ...
            'max_constraints', 100 ...
        );
        lbm = LBM(lbm_params);

        % Parametri SVR
        svr_params = struct(...
            'max_iter',        150, ...
            'kernel_function', kernel, ...
            'C',               1, ...
            'epsilon',         epsilon, ...
            'opt',             lbm ...
        );

        % SVR con e senza LBM
        svr_lbm = SVR(svr_params);
        svr_std = SVR(rmfield(svr_params, 'opt'));

        % Addestramento
        [~, f_values_std, ~, res] = svr_std.fit(X, y);
        pd_gap = abs(res.sol.itr.pobjval - res.sol.itr.dobjval);
        if pd_gap > 1e-5
            gaps(i) = inf;
            continue
        end

        [~, f_values_lbm, ~] = svr_lbm.fit(X, y);

        % Calcolo gap
        fmin_lbm = min(f_values_lbm);
        fmin_std = min(f_values_std);
        gap = abs(fmin_lbm - fmin_std) / abs(fmin_std);
        gaps(i) = gap;

    catch ME
        warning("Errore nella combinazione %d: %s", i, ME.message);
        fprintf("Sigma:   %.3f\n", sigma);
        fprintf("Theta:   %.3f\n", theta);
        fprintf("Epsilon: %.3f\n", epsilon);
        gaps(i) = Inf;
        error(ME.message)
    end
end

% Trova migliore combinazione
[~, best_idx] = min(gaps);
best_params = param_grid(best_idx, :);

fprintf("\n== MIGLIORI PARAMETRI TROVATI ==\n");
fprintf("Sigma:   %.3f\n", best_params(1));
fprintf("Theta:   %.3f\n", best_params(2));
fprintf("Epsilon: %.3f\n", best_params(3));
fprintf("Gap:     %.6f\n", gaps(best_idx));