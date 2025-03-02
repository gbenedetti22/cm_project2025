classdef LBM < handle    
    properties(Access=private)
        max_iter
        epsilon
        tol
        theta
        max_constraints
        qp_ratio
    end
    
    methods
        function obj = LBM(max_iter, epsilon, tol, theta, max_constraints, qp_ratio)
            if nargin < 1
                max_iter = 500;
            end
            if nargin < 2
                epsilon = 1e-6;
            end
            if nargin < 3
                tol = 1e-6;
            end
            if nargin < 4
                theta = 0.5;
            end
            if nargin < 5
                max_constraints = inf;
            end
            if nargin < 6
                qp_ratio = 10; % Default: usa quadprog nel 10% dei cicli
            end

            obj.max_iter = max_iter;
            obj.epsilon = epsilon;
            obj.tol = tol;
            obj.theta = theta;
            obj.max_constraints = max_constraints;
            obj.qp_ratio = qp_ratio;
        end
        function alpha = optimize(obj, K, y, C)
            n = length(y);
            z0 = zeros(2*n, 1);
            z = obj.project(z0, C);
            [f, g] = obj.svr_dual_function(z, K, y, obj.epsilon);
            f_best = f;
        
            bundle.z = z;
            bundle.f = f;
            bundle.g = g;
            bundle.K = K;  % Memorizza la matrice kernel nel bundle
            bundle.y = y;  % Memorizza i valori target nel bundle
        
            t = obj.tol;
            
            loading_bar = waitbar(0,'Computing LBM...');
            for iter = 1:obj.max_iter
                waitbar(iter/obj.max_iter, loading_bar, "Computing LBM (" + (iter) + "/" + obj.max_iter + ")");
        
                level = obj.theta * f + (1 - obj.theta) * f_best;
                z_new = obj.mp_solve(z, bundle, t, C, iter, obj.qp_ratio);
        
                step = z_new - z;
        
                [f_new, g_new] = obj.svr_dual_function(z_new, K, y, obj.epsilon);
                
                if f_new < f_best
                    f_best = f_new;
                end
                
                if size(bundle.z, 2) > obj.max_constraints
                    bundle.z = bundle.z(:, 2:end);
                    bundle.f = bundle.f(2:end);
                    bundle.g = bundle.g(:, 2:end);
                end
        
                bundle.z = [bundle.z, z_new];
                bundle.f = [bundle.f, f_new];
                bundle.g = [bundle.g, g_new];
                
                if f_new <= level
                    z = z_new;
                    f = f_new;
                else
                    t = t * 0.5;
                end
        
                if norm(step) < obj.tol
                    break;
                end
            end
        
            delete(loading_bar);
        
            alpha = z;
        end
        % function alpha = optimize(obj, K, y, C)
        %     n = length(y);
        %     z0 = zeros(2*n, 1);
        %     z = obj.project(z0, C);
        %     [f, g] = obj.svr_dual_function(z, K, y, obj.epsilon);
        %     f_best = f;
        % 
        %     bundle.z = z;
        %     bundle.f = f;
        %     bundle.g = g;
        % 
        %     t = obj.tol;
        % 
        %     loading_bar = waitbar(0,'Computing LBM...');
        %     for iter = 1:obj.max_iter
        %         waitbar(iter/obj.max_iter, loading_bar, "Computing LBM (" + (iter) + "/" + obj.max_iter + ")");
        % 
        %         level = obj.theta * f + (1 - obj.theta) * f_best;
        %         % z_new = obj.mp_solve(z, bundle, t, C);
        %         z_new = obj.mp_solve(z, bundle, t, C, iter, obj.qp_ratio);
        % 
        % 
        %         step = z_new - z;
        % 
        %         [f_new, g_new] = obj.svr_dual_function(z_new, K, y, obj.epsilon);
        % 
        %         if f_new < f_best
        %             f_best = f_new;
        %         end
        % 
        %         if size(bundle.z, 2) > obj.max_constraints
        %             bundle.z = bundle.z(:, 2:end);
        %             bundle.f = bundle.f(2:end);
        %             bundle.g = bundle.g(:, 2:end);
        %         end
        % 
        %         bundle.z = [bundle.z, z_new];
        %         bundle.f = [bundle.f, f_new];
        %         bundle.g = [bundle.g, g_new];
        % 
        %         if f_new <= level
        %             z = z_new;
        %             f = f_new;
        %         else
        %             t = t * 0.5;
        %         end
        % 
        %         if norm(step) < obj.tol
        %             break;
        %         end
        % 
        %     end
        % 
        %     delete(loading_bar);
        % 
        %     alpha = z;
        % end
    end

    methods (Access=private)
        function [f, g] = svr_dual_function(~, z, K, y, epsilon)
            n = length(y);
            alpha = z(1:n);
            alphaStar = z(n+1:end);
            
            % Create the block matrix representation to match quadprog
            H = [K, -K; -K, K];
            f_vec = [epsilon + y; epsilon - y];
            
            % Compute objective value (note the negative sign to match quadprog)
            diffAlpha = [alpha; alphaStar];
            f = -(-0.5 * diffAlpha' * H * diffAlpha + f_vec' * diffAlpha);
            
            % Gradient (again with sign adjustment)
            g = -(-H * diffAlpha + f_vec);
        end

        function z_proj = project(~, z, C)
            n = numel(z)/2;
            
            % First apply box constraints
            alpha = min(max(z(1:n), 0), C);
            alphaStar = min(max(z(n+1:end), 0), C);
            
            % Then enforce the equality constraint sum(alpha) = sum(alphaStar)
            diff = (sum(alpha) - sum(alphaStar)) / (2*n);
            alpha = min(max(alpha - diff, 0), C);
            alphaStar = min(max(alphaStar + diff, 0), C);
            
            z_proj = [alpha; alphaStar];
        end

        % function z_opt = mp_solve(~, z_current, bundle, t, C)
        %     n2 = length(z_current);
        %     n = n2/2;
        %     m = length(bundle.f);
        % 
        %     H = sparse(1:n2, 1:n2, 1/t, n2, n2);
        %     H = blkdiag(H, 0);
        % 
        %     f = sparse([-(1/t) * z_current; 1]);
        % 
        %     A_bundle = sparse([bundle.g' -ones(m,1)]);
        %     b_bundle = sparse(sum(bundle.g .* bundle.z, 1)' - bundle.f');
        % 
        %     Aeq = sparse([ones(1,n), -ones(1,n), 0]);
        %     beq = 0;
        % 
        %     lb = sparse([zeros(n2,1); -inf]);
        %     ub = sparse([C * ones(n2,1);  inf]);
        % 
        %     if exist('mosekopt', 'file') == 3
        %         options = mskoptimset('Display', 'off');
        %     else
        %         options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');
        %     end
        % 
        %     [z_sol, ~, exitflag] = quadprog(H, f, A_bundle, b_bundle, Aeq, beq, lb, ub, [], options);
        % 
        %     if exitflag <= 0
        %         warning('quadprog non ha trovato una soluzione valida. Manteniamo la soluzione corrente.');
        %         z_opt = z_current;
        %     else
        %         z_opt = z_sol(1:n2);
        %     end
        % end
        function z_opt = mp_solve(obj, z_current, bundle, t, C, iter, qp_ratio)
            n2 = length(z_current);
            n = n2 / 2;
            m = length(bundle.f);
            
            % Calcola se in questo iter si deve usare quadprog in base a qp_ratio
            use_quadprog = (mod(iter, round(100 / qp_ratio)) == 0);
            
            % Se qp_ratio è 100, allora usa sempre quadprog
            if qp_ratio == 100
                use_quadprog = true;
            end
        
            % Se qp_ratio è 0, usa sempre subgradiente
            if qp_ratio == 0
                use_quadprog = false;
            end
        
            % Inizializza soluzione
            z_opt = z_current;
            
            if use_quadprog
                % Usa quadprog per la soluzione più precisa
                H = sparse(1:n2, 1:n2, 1/t, n2, n2);
                H = blkdiag(H, 0);
                f = sparse([-(1/t) * z_current; 1]);
        
                A_bundle = sparse([bundle.g' -ones(m,1)]);
                b_bundle = sparse(sum(bundle.g .* bundle.z, 1)' - bundle.f');
        
                Aeq = sparse([ones(1,n), -ones(1,n), 0]);
                beq = 0;
        
                lb = sparse([zeros(n2,1); -inf]);
                ub = sparse([C * ones(n2,1); inf]);
        
                options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');
        
                [z_sol, ~, exitflag] = quadprog(H, f, A_bundle, b_bundle, Aeq, beq, lb, ub, [], options);
        
                if exitflag > 0
                    z_opt = z_sol(1:n2);
                else
                    % Se quadprog fallisce, usa subgradiente come backup
                    z_opt = subgradient_step(obj, z_current, bundle, t, C);
                end
            else
                % Usa solo subgradiente
                z_opt = subgradient_step(obj, z_current, bundle, t, C);
            end
        end

        % Implementazione del passo di subgradiente con condizione di Armijo
        function z_sg = subgradient_step(obj, z_current, bundle, t, C)
            
            % Usa la media pesata dei subgradienti nel bundle
            bundle_size = size(bundle.g, 2);
            if bundle_size > 0
                % Pesi basati sulla recenza - i subgradienti più recenti hanno peso maggiore
                weights = exp(linspace(-3, 0, bundle_size));
                weights = weights / sum(weights);
                
                % Calcola subgradiente medio pesato
                g_avg = zeros(size(bundle.g, 1), 1);
                for i = 1:bundle_size
                    g_avg = g_avg + weights(i) * bundle.g(:, i);
                end
                
                % Calcola la direzione di discesa
                d = -g_avg;
                
                % Valutazione della funzione nel punto corrente
                [f_current, ~] = obj.svr_dual_function(z_current, bundle.K, bundle.y, obj.epsilon);
                
                % Parametri per la condizione di Armijo
                alpha = 1.0;  % Step size iniziale
                beta = 0.5;   % Fattore di riduzione dello step size
                sigma = 0.1;  % Parametro per condizione di sufficiente decremento
                
                % Massimo numero di iterazioni per la ricerca in linea
                max_line_search = 10;
                
                for i = 1:max_line_search
                    % Calcola punto candidato
                    z_new = z_current + alpha * d;
                    
                    % Proietta sulla regione ammissibile
                    z_candidate = obj.project(z_new, C);
                    
                    % Valuta la funzione nel punto candidato
                    [f_new, ~] = obj.svr_dual_function(z_candidate, bundle.K, bundle.y, obj.epsilon);
                    
                    % Verifica la condizione di Armijo (sufficiente decremento)
                    if f_new <= f_current + sigma * g_avg' * (z_candidate - z_current)
                        z_sg = z_candidate;
                        return;
                    end
                    
                    % Riduci lo step size
                    alpha = beta * alpha;
                end
                
                % Se non è stata soddisfatta la condizione di Armijo, usa uno step piccolo
                step_size = t / (norm(g_avg) + 1e-8) * 0.1;
                z_new = z_current - step_size * g_avg;
                z_sg = obj.project(z_new, C);
            else
                % Se il bundle è vuoto, mantieni il punto corrente
                z_sg = z_current;
            end
        end

        % function z_sg = subgradient_step(obj, z_current, bundle, t, C)
        %     % Use weighted average of subgradients in the bundle
        %     bundle_size = size(bundle.g, 2);
        %     if bundle_size > 0
        %         % Use recency weighting - more recent subgradients get higher weight
        %         weights = exp(linspace(-3, 0, bundle_size));
        %         weights = weights / sum(weights);
        % 
        %         % Compute weighted average subgradient
        %         g_avg = zeros(size(bundle.g, 1), 1);
        %         for i = 1:bundle_size
        %             g_avg = g_avg + weights(i) * bundle.g(:, i);
        %         end
        % 
        %         % Adaptive step size based on bundle information
        %         step_size = t / (norm(g_avg) + 1e-8);
        % 
        %         % Take step with momentum
        %         persistent prev_step;
        %         if isempty(prev_step)
        %             prev_step = zeros(size(z_current));
        %         end
        % 
        %         momentum = 0.3; % Add 30% of previous step
        %         z_new = z_current - step_size * g_avg + momentum * prev_step;
        %         prev_step = z_new - z_current;
        % 
        %         % Project onto the feasible region
        %         z_sg = obj.project(z_new, C);
        %     else
        %         z_sg = z_current;
        %     end
        % end

        % % Helper function for subgradient steps
        % function z_sg = subgradient_step(obj, z_current, bundle, t, C)
        %     % Use the latest subgradient (most recent point in the bundle)
        %     latest_idx = size(bundle.g, 2);
        %     if latest_idx > 0
        %         g = bundle.g(:, latest_idx);
        %     else
        %         % If bundle is empty, use a zero gradient (shouldn't happen in practice)
        %         g = zeros(size(z_current));
        %     end
        % 
        %     % Calculate step size - adaptive based on iteration progress
        %     step_size = t / (norm(g) + 1e-8);
        % 
        %     % Take a step in the negative gradient direction
        %     z_new = z_current - step_size * g;
        % 
        %     % Project onto the feasible region
        %     z_sg = obj.project(z_new, C);
        % end


    end
end