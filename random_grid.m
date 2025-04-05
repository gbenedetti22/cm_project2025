%% Random Grid Search con valori predefiniti per parametri e monitoraggio dei progressi
% Pulisce l'ambiente di lavoro
clear; clc; close all;

%% Caricamento e preprocessing dei dati
[X, y] = training_data("abalone");
X = zscore(X);

%% Parametri costanti dello SVR
C = 1;                          % BoxConstraint
global_epsilon = 0.05;          % ε usato sia per lo SVR che per LBM
tol_default = 1e-5;             % Tolleranza usata in SVR (default)

%% Vettori dei possibili valori per i parametri della LBM
max_iter_vals = [50, 100];           % Numero massimo di iterazioni
epsilon_vals  = [1e-4, 1e-6, 1e-5];    % Parametro ε per LBM (se vuoi mantenerlo diverso, puoi usare un altro vettore o usare global_epsilon)
tol_vals      = [1e-7, 1e-6, 1e-8];    % Tolleranza per LBM
theta_vals    = [0.1, 0.05, 0.01];       % Valori di theta
max_constraints_vals = [50, 100];             % Numero massimo di vincoli (fisso)

% Nuovo vettore per λ (lambda)
lambda_vals = [0.7, 0.5, 0.6];        % Valori iniziali di lambda

%% Vettore per il parametro del kernel RBF
sigma_vals = [0.9, 0.85, 0.8];

%% Numero di iterazioni della ricerca randomica
numTrials = 10;  

%% Variabili per salvare i risultati
bestMSE = inf;
bestParams = [];
results = struct('trial', {}, 'mse', {}, 'lbm_params', {}, 'sigma', {}, 'trialTime', {});

%% Random Grid Search
for trial = 1:numTrials
    trialStart = tic;  

    % Selezione randomica di un valore per ciascun parametro
    max_iter = max_iter_vals(randi(numel(max_iter_vals)));
    epsilon_LBM = epsilon_vals(randi(numel(epsilon_vals)));  
    tol_val = tol_vals(randi(numel(tol_vals)));
    max_constraints_val = max_constraints_vals(randi(numel(max_constraints_vals)));
    theta_val = theta_vals(randi(numel(theta_vals)));
    lambda_val = lambda_vals(randi(numel(lambda_vals)));      % nuovo parametro λ
    sigma = sigma_vals(randi(numel(sigma_vals)));
    
    % Visualizza i parametri della configurazione corrente
    fprintf('Trial %d:\n', trial);
    fprintf('  max_iter = %d\n  epsilon_LBM = %.2e\n  tol = %.2e\n  theta = %.2f\n  lambda = %.2f\n  max_constraints = %d\n  sigma = %.2f\n',...
        max_iter, epsilon_LBM, tol_val, theta_val, lambda_val, max_constraints_val, sigma);
    
    % Configurazione dei parametri LBM (includi lambda)
    lbm_params = struct(...
        'max_iter', max_iter, ...
        'epsilon', epsilon_LBM, ...      
        'tol', tol_val, ...
        'theta', theta_val, ...
        'max_constraints', max_constraints_val, ...
        'lambda', lambda_val);           % nuovo iperparametro
    
    lbm = LBM(lbm_params);
    
    % Configurazione del kernel e dei parametri dello SVR
    kernel = RBFKernel(sigma);
    svr_params = struct(...
        'kernel_function', kernel, ...
        'C', C, ...
        'epsilon', global_epsilon, ...   
        'opt', lbm, ...
        'tol', tol_default);
    
    svr_model = SVR(svr_params);
    
    % Addestramento del modello e calcolo del MSE
    try
        [alpha, f_hist] = svr_model.fit(X, y);
        y_pred = svr_model.predict(X);
        mse_val = mse(y_pred, y);
    catch ME
        fprintf('  Trial %d fallito: %s\n', trial, ME.message);
        mse_val = inf;
    end
    
    % Tempo impiegato per il trial corrente
    trialTime = toc(trialStart);
    
    % Salva i risultati del trial
    results(trial).trial = trial;
    results(trial).mse = mse_val;
    results(trial).lbm_params = lbm_params;
    results(trial).sigma = sigma;
    results(trial).trialTime = trialTime;
    
    % Stampa il risultato del trial
    fprintf('  MSE = %.4f | Tempo = %.2f s\n\n', mse_val, trialTime);
    
    % Aggiorna la migliore configurazione
    if mse_val < bestMSE
        bestMSE = mse_val;
        bestParams = struct('lbm_params', lbm_params, 'sigma', sigma);
    end

end


%% Risultati finali
fprintf('\n=== Miglior Risultato ===\n');
fprintf('MSE migliore: %.4f\n', bestMSE);
disp('Parametri migliori:');
disp(bestParams);
