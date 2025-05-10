clc; clear; close all;

% Carico e normalizzo i dati
[X, y] = training_data("red_wine");
X = zscore(X);

% Valore di riferimento ottenuto dall'oracle (fmincon)
f_val = -481.2514;  % Assicurati che questo valore sia corretto

% Parametri fissi
maxIter = 150;
lbm_tol = 1e-8;
lbm_max_constr = 100;
C = 3;
epsilon = 0.01;
sigma = 0.7;

% Valori di theta da testare
theta_values = [0.4, 0.5, 0.6, 0.7];
N = length(theta_values);

% Inizializza il pool parallelo (se non è già attivo)
if isempty(gcp('nocreate'))
    parpool;
end

% Prealloca strutture per i risultati
gap_rels = nan(N,1);
times = nan(N,1);
thetas_tested = nan(N,1);

parfor i = 1:N
    theta = theta_values(i);
    thetas_tested(i) = theta;
    
    lbm_params = struct(...
        'tol', lbm_tol, ...
        'theta', theta, ...
        'max_constraints', lbm_max_constr ...
    );
    lbm = LBM(lbm_params);
    
    svr_params = struct(...
        'max_iter', maxIter, ...
        'kernel_function', RBFKernel(sigma), ...
        'C', C, ...
        'epsilon', epsilon, ...
        'opt', lbm ...
    );
    svr_lbm = SVR(svr_params);
    
    t_start = tic;
    try
        [x_lbm, f_values_lbm] = svr_lbm.fit(X, y);
        elapsed_time = toc(t_start);
        
        if ~isempty(f_values_lbm)
            min_f_lbm = min(f_values_lbm);
            gap_rel = abs(min_f_lbm - f_val) / (abs(f_val) + eps);
        else
            gap_rel = Inf;
        end
        
        gap_rels(i) = gap_rel;
        times(i) = elapsed_time;
        
        fprintf('θ: %.1f | Gap: %.4e | Time: %.2f s\n', ...
                theta, gap_rel, elapsed_time);
    catch ME
        elapsed_time = toc(t_start);
        gap_rels(i) = Inf;
        times(i) = elapsed_time;
        fprintf('Errore con θ=%.1f: %s\n', theta, ME.message);
    end
end

% Trova la migliore configurazione
[best_gap, idx] = min(gap_rels);
best_theta = thetas_tested(idx);

fprintf('\nMiglior θ trovato: %.2f\n', best_theta);
fprintf("Gap relativo: %.4e\n", best_gap);
fprintf("Tempo di esecuzione: %.2f s\n", times(idx));

% Ordina i risultati per theta crescente
[thetas_sorted, sort_idx] = sort(thetas_tested);
gap_rels_sorted = gap_rels(sort_idx);
times_sorted = times(sort_idx);

% Plot risultati
figure;
subplot(2,1,1);
plot(thetas_sorted, gap_rels_sorted, '-o');
xlabel('θ'); ylabel('Gap relativo');
title('Performance al variare di θ');
grid on;

subplot(2,1,2);
plot(thetas_sorted, times_sorted, '-o');
xlabel('θ'); ylabel('Tempo (s)');
title('Tempo di esecuzione al variare di θ');
grid on;