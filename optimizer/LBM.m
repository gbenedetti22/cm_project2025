classdef LBM < handle    
    properties(Access=private)
        max_iter     
        epsilon      
        tol          
        theta        
        max_constraints
        lambda
    end
    
    methods
        function obj = LBM(params)
            if ~isstruct(params)
                error('LBM richiede parametri in formato struct');
            end
            obj.init_parameters(params);
        end
        
        function init_parameters(obj, params)
            % Inizializza parametri con valori di default
            default_params = struct(...
                'max_iter', 500, ...
                'epsilon', 1e-6, ...
                'tol', 1e-4, ...
                'theta', 0.5, ...
                'max_constraints', 100, ...
                'lambda', 0.1);
            
            fields = fieldnames(default_params);
            for i = 1:length(fields)
                field = fields{i};
                if isfield(params, field)
                    obj.(field) = params.(field); % Assegna i parametri dall'input
                else
                    obj.(field) = default_params.(field); % Usa i default
                end
                fprintf('Parametro %s: %s\n', field, mat2str(obj.(field)));
            end
        end
        
        function [alpha, f_values] = optimize(obj, K, y, C, epsilon)
            n = length(y);
            z = zeros(n, 1);
            [f, g] = obj.svr_dual_function(z, K, y, epsilon);
            f_best = f;
            
            bundle.z = z;
            bundle.f = f;
            bundle.g = g;
            
            % Utilizza il valore di lambda passato come iperparametro
            current_lambda = obj.lambda;  
            
            f_values = zeros(obj.max_iter, 1);
            f_values(1) = f;
            
            for iter = 1:obj.max_iter 
                level = obj.theta * f + (1 - obj.theta) * f_best;
                z_new = obj.mp_solve(z, bundle, level, C);
                step = z_new - z;
                
                % Aggiornamento convesso usando current_lambda
                z = (1 - current_lambda) * z + current_lambda * z_new;
                
                % Calcola nuovo valore e gradiente
                [f_new, g_new] = obj.svr_dual_function(z, K, y, epsilon);
                f_values(iter) = f_new;
                
                % Aggiorna miglior valore
                if f_new < f_best
                    f_best = f_new;
                end
                
                % Gestione del bundle
                if size(bundle.z, 2) >= obj.max_constraints
                    bundle.z = bundle.z(:, 2:end);
                    bundle.f = bundle.f(2:end);
                    bundle.g = bundle.g(:, 2:end);
                end
                bundle.z = [bundle.z, z_new];
                bundle.f = [bundle.f, f_new];
                bundle.g = [bundle.g, g_new];
                
                % Se desideri aggiornare dinamicamente lambda, decommenta
                % current_lambda = min(1, current_lambda + 0.05);
                % obj.theta = min(0.99, obj.theta + 0.01);
                
                % Criterio di arresto
                if norm(step) < obj.tol
                    break;
                end
            end
            alpha = z;
            f_values = f_values(1:iter);
        end
    end
    
    methods (Access=private)
        function [f, g] = svr_dual_function(~, x, K, y, epsilon)
            f = 0.5 * x' * (K * x) + epsilon * sum(abs(x)) - y' * x;
            g = K * x + epsilon * sign(x) - y;
            
            zero_indices = (x == 0);
            g(zero_indices) = -y(zero_indices); 
        end
        
        function alpha_opt = mp_solve(~, alpha_hat, bundle, f_level, C)
            n = length(alpha_hat);
            m = length(bundle.f);
            
            H = blkdiag(speye(n), 0);
            f = [-alpha_hat; 0];
            
            A = sparse([bundle.g' -ones(m, 1)]);
            b = sum(bundle.g .* bundle.z, 1)' - bundle.f';
            
            Aeq = sparse([ones(1, n) 0]);
            beq = 0;
            
            lb = [-C * ones(n, 1); -inf];
            ub = [C * ones(n, 1); f_level];
            
            options = optimoptions('quadprog', 'Display', 'off');
            sol = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], options);
            
            if isempty(sol)
                alpha_opt = alpha_hat;
            else
                alpha_opt = sol(1:n);
            end
        end
    end
end
