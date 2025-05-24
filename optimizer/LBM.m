% Level Bundle Method implementation using lower bound estimation and MOSEK
% as Solver

% Made by: Benedetti Gabriele and Federico Bonaccorsi
classdef LBM < handle    
    properties(Access=private)
        tol
        theta
        max_constraints
        logger
        log
    end
    
    methods
        function obj = LBM(params)
            if ~isstruct(params)
                error('Input must be a valid struct');
            end

            default_params = struct(...
                'tol', 1e-6,...
                'theta', 0.5,...
                'log', "summary",...
                'max_constraints', inf....
            );

            all_fields = fieldnames(default_params);
            for i = 1:length(all_fields)
                field = all_fields{i};
                if isfield(params, field)
                    obj.(field) = params.(field);
                else
                    obj.(field) = default_params.(field);
                end
            end

            obj.logger = Logger(obj.log);
        end
        
        function [alpha, history] = optimize(obj, K, y, C, max_iter, epsilon)
            % Input:
            %   K - kernel matrix
            %   y - target values
            %   C - bounding parameter
            %   max_iter - maximum number of iterations
            %   epsilon - regularization parameter
            % Output:
            %   alpha - computed optimal solution
            %   history - optimization history

            %% Initial steps
            n = length(y);
            history = OptHistory(max_iter);

            alpha = zeros(n, 1); % we start from zero
            [f, g] = obj.svr_dual_function(alpha, K, y, epsilon);
            f_best = f;
            
            bundle.alpha = alpha;
            bundle.f = f;
            bundle.g = g;

            gaps = nan(max_iter, 1);
            
            tic
            for iter = 1:max_iter
                %% Lower bound estimation
                lb = obj.compute_lower_bound(bundle, C);
                level = lb + obj.theta * (f_best - lb);
                gap = abs(f_best - lb) / abs(f_best);
                
                %% Solve the master problem and get the new (best) point
                alpha_new = obj.mp_solve(alpha, bundle, level, C);
                alpha = alpha_new;

                [f_new, g_new] = obj.svr_dual_function(alpha_new, K, y, epsilon);
                
                if f_new < f_best
                    f_best = f_new;
                end

                history.f_values(iter) = f_new;
                history.f_times(iter) = toc;
                gaps(iter)=gap;

                obj.logger.log(iter, f_new, g_new, gap, f_best);
                
                %% % Update bundle
                if size(bundle.alpha, 2) > obj.max_constraints
                    bundle.alpha = bundle.alpha(:, 2:end);
                    bundle.f = bundle.f(2:end);
                    bundle.g = bundle.g(:, 2:end);
                end

                bundle.alpha = [bundle.alpha, alpha_new];
                bundle.f = [bundle.f, f_new];
                bundle.g = [bundle.g, g_new];
                
                %% Check for convergence (distance between upper and lower bound)
                if gap < obj.tol
                    break;
                end

            end
            history.f_times = history.f_times(~isnan(history.f_times));

            obj.logger.summary(iter, min(history.f_values), min(gaps), min(history.f_times));
        end
    end

    methods (Access=private)
        
        function [f, g] = svr_dual_function(~, x, K, y, epsilon)
            % SVR Dual function with absolute value 
            f = 0.5 * x' * (K * x) + epsilon * sum(abs(x)) - y' * x;

            % subgradient
            g = K * x + epsilon * sign(x) - y;
        end

        function lb = compute_lower_bound(~, bundle, C)
            % Solves a LP problem to compute a lower bound
            %
            % Inputs:
            %   bundle - current bundle
            %   C - box constraint
            %
            % Output:
            %   lb: computed lower bound

            %% Inizialization
            [n, m] = size(bundle.alpha);

            %% Reformulating problem for linprog
            % since we only have t
            f = [zeros(n, 1); 1];

            %% Constraints
            % Inequality constraints (f_i + <grad_i, alpha - alpha_i> ≤ t)
            % Here: A = <grad_i, alpha> - t and b = <grad_i, alpha_i> - f_i
            A = [bundle.g', -ones(m, 1)];
            b = sum(bundle.g .* bundle.alpha, 1)' -bundle.f';

            % Equality Constraints (sum(alpha_i) = 0)
            Aeq = [ones(1, n), 0];
            beq = 0;

            % Box contraints (-C ≤ alpha_i ≤ C)
            lb  = [-C * ones(n, 1); -inf];
            ub  = [ C * ones(n, 1);  inf];

            %% Solve LP using linprog
            options = optimoptions('linprog', 'Display', 'off');
            [~, lb, exitflag] = linprog(f, A, b, Aeq, beq, lb, ub, options);

            if exitflag <= 0
                warning('Error while computing lower bound');
                lb = min(bundle.f);
            end
        end

        
        function alpha_opt = mp_solve(~, alpha_hat, bundle, f_level, C)
            % Solves the Master Problem of the Level Bundle Method
            %
            % Inputs:
            %   alpha_hat - current solution
            %   bundle - struct bundle
            %   f_level - current level used as upper bound
            %   C - box constraint
            %
            % Output:
            %   alpha_opt - optimized solution
            
            %% Inizialization
            n = length(alpha_hat);
            m = length(bundle.f);

            %% Reformulating problem for Quadprog
            % quadprog standard form: min 1/2 x'Hx + f'x
            % Here: H = I and f = -alpha_hat
            H = blkdiag(speye(n), 0); % padding added for level handling (t) 
            f = sparse([-alpha_hat; 0]);

            %% Constraints
            % Bundle cuts constraint
            % quadprog standard form: Ax <= b
            % here: A = [g_i' -1] and b = sum(g_i * alpha_i) - f_i
            A = sparse([bundle.g' -ones(m, 1)]);
            b = sparse(sum(bundle.g .* bundle.alpha, 1)' - bundle.f');

            % Equality constraint
            % sum(alpha_i) = 0
            Aeq = sparse([ones(1, n) 0]);
            beq = 0;

            % Bounds constraints (lb ≤ x ≤ ub)
            % -C <= alpha_i <= C
            % -inf <= t <= f_level
            lb = sparse([-C * ones(n, 1); -inf]);
            ub = sparse([C * ones(n, 1); f_level]);

            %% Solve QP using quadprog (MOSEK)
            options = mskoptimset('Display', 'off');

            [sol, ~, exitflag] = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], options);

            if exitflag <= 0
                warning("best solution not found, keeping prev...");
                alpha_opt = alpha_hat;
            else
                alpha_opt = sol(1:n);
            end
        end
    end
end