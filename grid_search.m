
optimize_lbm_params()

function optimize_lbm_params()
    % Avvia il pool parallelo
    if isempty(gcp('nocreate'))
        parpool('local', feature('numcores'));
    end
    
    % Caricamento e preparazione dati
    [X, y] = training_data("airfoil");
    X = zscore(X);
    
    % Parametri fissi
    sigma = 0.6;
    C = 1;
    max_iter = 90;
    max_constraints = 60;
    tol = 1e-2;
    
    % Range di ricerca ridotti
    lr_values = logspace(-7, -6, 5);           
    momentum_values = [0.3, 0.4, 0.5, 0.6];        
    theta_values = linspace(0.5, 0.8, 4);     
    epsilon_values = 0.02;             % Solo estremi
    scale_values = logspace(-5, -3, 4);       
    
    param_grid = combvec(lr_values, momentum_values, theta_values, epsilon_values, scale_values)';
    total_comb = size(param_grid, 1);
    
    % Inizializza barra di avanzamento
    h = waitbar(0, 'Avanzamento grid search...', 'Name','Ricerca Parametri');
    D = parallel.pool.DataQueue;
    afterEach(D, @(~) progress_update(h, total_comb));
    
    % Risultati
    results = cell(total_comb, 2);
    
    % Ricerca parallela
    parfor i = 1:total_comb
        % Estrai parametri
        params = param_grid(i,:);
        [lr, momentum, theta, epsilon, scale] = deal(params(1), params(2), params(3), params(4), params(5));
        
        % Configurazione LBM
        lbm_params = struct(...
            'tol', tol, ...
            'theta', theta, ...
            'lr', lr, ...
            'momentum', momentum, ...
            'scale_factor', scale, ...
            'max_constraints', max_constraints);
        
        % Configurazione SVR
        svr_params = struct(...
            'max_iter', max_iter, ...
            'kernel_function', RBFKernel(sigma), ...
            'C', C, ...
            'epsilon', epsilon, ...
            'opt', LBM(lbm_params));
        
        % Valutazione
        try
            svr = SVR(svr_params);
            svr.fit(X, y);
            results(i,:) = {mse(svr.predict(X), y), lbm_params};
        catch
            results(i,:) = {inf, struct()};
        end
        
        send(D, i); % Aggiorna barra
    end
    
    delete(h); % Chiudi barra
    
    % Risultati finali
    [best_mse, idx] = min([results{:,1}]);
    best_params = results{idx,2};
    
    fprintf('\n=== Risultati Ottimali ===\n');
    fprintf('MSE: %.5f\n', best_mse);
    disp('Parametri:');
    disp(best_params);
    
    save('results_light.mat', 'results', 'best_params');
end

function progress_update(h, total)
    persistent count start_time
    if isempty(count)
        count = 0;
        start_time = tic;
    end
    count = count + 1;
    
    elapsed = toc(start_time);
    remaining = (elapsed/count)*(total - count);
    
    waitbar(count/total, h, sprintf('Completato: %d/%d\nTempo: %s\nRimanente: %s',...
        count, total, ...
        datestr(seconds(elapsed),'MM:SS'), ...
        datestr(seconds(remaining),'MM:SS')));
end