

optimized_airfoil_search()
function optimized_airfoil_search()
    % Avvia il pool parallelo
    if isempty(gcp('nocreate'))
        parpool('local', feature('numcores'));
    end
    
    rng(42); % Seed fisso per riproducibilità
    
    % Caricamento e preparazione dati
    [X, y] = training_data("airfoil");
    X = zscore(X);
    
    % 1. Calcolo baseline con parametri espliciti
    fprintf('\n=== Calcolo Baseline ===\n');
    svr_baseline = SVR(struct(...
        'kernel_function', RBFKernel(0.6),...
        'C', 1,...
        'epsilon', 0.02,...
        'max_iter', 90));
    
    svr_baseline.fit(X, y);
    baseline_mse = mse(svr_baseline.predict(X), y);
    fprintf('Baseline MSE: %.6f\n', baseline_mse);
    
    % 2. Fase 1: Broad Random Search con parametri ottimizzati
    fprintf('\n=== Fase 1: Broad Random Search ===\n');
    lbm_ranges = struct(...
        'theta',    [0.45, 0.56],...
        'lr',       [1e-7, 2e-5],...
        'momentum', [0.3, 0.8],...
        'scale',    [1e-5, 1e-4]);
    
    [best_params, best_mse] = parallel_random_search(...
        X, y, baseline_mse, lbm_ranges,...
        'MaxIter', 50, 'PruneFactor', 1.2, 'MaxIterLBM', 30);
    
    % 3. Fase 2: Adaptive Grid Search
    fprintf('\n=== Fase 2: Adaptive Grid Search ===\n');
    final_params = smart_grid_search(...
        X, y, best_params, lbm_ranges, baseline_mse,...
        'GridPoints', 3, 'DeltaPercent', 0.15, 'MaxIterLBM', 50);
    
    % 4. Validazione finale
    validate_final_results(X, y, final_params, baseline_mse);
end

function [best_params, best_mse] = parallel_random_search(X, y, baseline_mse, lbm_ranges, varargin)
    % Parametri di default
    defaults = struct('MaxIter', 50, 'PruneFactor', 1.2, 'MaxIterLBM', 30);
    opts = merge_structs(defaults, struct(varargin{:}));
    
    results = cell(opts.MaxIter, 2);
    
    parfor i = 1:opts.MaxIter
        params = struct(...
            'theta',    rand_in_range(lbm_ranges.theta),...
            'lr',       rand_in_range(lbm_ranges.lr),...
            'momentum', rand_in_range(lbm_ranges.momentum),...
            'scale',    rand_in_range(lbm_ranges.scale),...
            'tol',      1e-3,...
            'max_constraints', 60);
        
        try
            svr_params = struct(...
                'kernel_function', RBFKernel(0.6),...
                'C', 1,...
                'epsilon', 0.02,...
                'max_iter', opts.MaxIterLBM,...
                'opt', LBM(params));
            
            svr = SVR(svr_params);
            svr.fit(X, y);
            mse_val = mse(svr.predict(X), y);
            
            if mse_val < baseline_mse * opts.PruneFactor
                results(i,:) = {mse_val, params};
            else
                results(i,:) = {Inf, struct()};
            end
        catch
            results(i,:) = {Inf, struct()};
        end
    end
    
    % Selezione migliori parametri
    all_mse = [results{:,1}];
    valid_idx = find(all_mse < Inf);
    
    if isempty(valid_idx)
        error('Nessuna configurazione valida trovata nella fase 1');
    end
    
    [best_mse, idx] = min(all_mse(valid_idx));
    best_params = results{valid_idx(idx),2};
end

function final_params = smart_grid_search(X, y, init_params, ranges, baseline_mse, varargin)
    % Parametri di default
    defaults = struct('GridPoints', 3, 'DeltaPercent', 0.15, 'MaxIterLBM', 50);
    opts = merge_structs(defaults, struct(varargin{:}));
    
    % Generazione grid
    [theta_grid, lr_grid, mom_grid, scale_grid] = generate_grid(init_params, ranges, opts);
    combinations = allcomb(theta_grid, lr_grid, mom_grid, scale_grid);
    
    % Valutazione parallela
    results = cell(size(combinations, 1), 2);
    
    parfor i = 1:size(combinations, 1)
        params = struct(...
            'theta',    combinations(i,1),...
            'lr',       combinations(i,2),...
            'momentum', combinations(i,3),...
            'scale',    combinations(i,4),...
            'tol',      1e-3,...
            'max_constraints', 60);
        
        try
            svr_params = struct(...
                'kernel_function', RBFKernel(0.6),...
                'C', 1,...
                'epsilon', 0.02,...
                'max_iter', opts.MaxIterLBM,...
                'opt', LBM(params));
            
            svr = SVR(svr_params);
            svr.fit(X, y);
            mse_val = mse(svr.predict(X), y);
            
            results(i,:) = {mse_val, params};
        catch
            results(i,:) = {Inf, struct()};
        end
    end
    
    % Selezione finale
    [~, idx] = min([results{:,1}]);
    final_params = results{idx,2};
end

function validate_final_results(X, y, params, baseline_mse)
    fprintf('\n=== PARAMETRI FINALI ===\n');
    disp(params)
    
    % Validazione con iterazioni complete
    svr_params = struct(...
        'kernel_function', RBFKernel(0.6),...
        'C', 1,...
        'epsilon', 0.02,...
        'max_iter', 90,...
        'opt', LBM(params));
    
    tic;
    svr = SVR(svr_params);
    svr.fit(X, y);
    final_mse = mse(svr.predict(X), y);
    time = toc;
    
    fprintf('\nBaseline MSE: %.6f\n', baseline_mse);
    fprintf('Final MSE:    %.6f\n', final_mse);
    fprintf('Differenza:   %.6f\n', abs(baseline_mse - final_mse));
    fprintf('Tempo esecuzione: %.2f secondi\n', time);
end

%% Funzioni di supporto
function [theta, lr, momentum, scale] = generate_grid(init_params, ranges, opts)
    % Funzioni helper per la generazione della grid
    theta = linspace_clamped(init_params.theta, opts.DeltaPercent, ranges.theta, opts.GridPoints);
    lr = logspace_clamped(init_params.lr, opts.DeltaPercent, ranges.lr, opts.GridPoints);
    momentum = linspace_clamped(init_params.momentum, 0.05, ranges.momentum, 2);
    scale = logspace_clamped(init_params.scale, opts.DeltaPercent, ranges.scale, opts.GridPoints);
end

function val = rand_in_range(range)
    val = range(1) + (range(2) - range(1)) * rand();
end

function arr = linspace_clamped(center, delta_percent, range, n)
    delta = center * delta_percent;
    lower = max(center - delta, range(1));
    upper = min(center + delta, range(2));
    arr = linspace(lower, upper, n);
end

function arr = logspace_clamped(center, delta_percent, range, n)
    delta = center * delta_percent;
    lower = max(center - delta, range(1));
    upper = min(center + delta, range(2));
    arr = exp(linspace(log(lower), log(upper), n));
end

function s = merge_structs(varargin)
    s = struct();
    for i = 1:nargin
        fields = fieldnames(varargin{i});
        for j = 1:length(fields)
            s.(fields{j}) = varargin{i}.(fields{j});
        end
    end
end

function c = allcomb(varargin)
    [c{1:nargin}] = ndgrid(varargin{:});
    c = reshape(cat(nargin+1, c{:}), [], nargin);
end