classdef LBM < handle    
    properties(Access=private)
        tol
        theta
        max_constraints
        lr
        momentum
        scale_factor
    end
    
    methods
        function obj = LBM(params)
            if ~isstruct(params)
                error('Input must be a valid struct');
            end

            default_params = struct(...
                'tol', 1e-6,...
                'theta', 0.5,...
                'lr', 1e-6,...
                'momentum', 0.6,...
                'scale_factor', 1e-8,...
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
        end
        
        function [alpha, f_values, f_times] = optimize(obj, K, y, C, max_iter, epsilon)
            n = length(y);
            z = zeros(n, 1);
            [f, g] = obj.svr_dual_function(z, K, y, epsilon);
            f_best = f;

            bundle.z = z;
            bundle.f = f;
            bundle.g = g;

            f_values = nan(max_iter, 1);
            f_times = nan(max_iter, 1);
            
            % h = animatedline('LineStyle','-', 'Marker','none', 'LineWidth', 2);
            % tic
            for iter = 1:max_iter
                lb = obj.compute_lower_bound(bundle, C);
                level = lb + obj.theta * (f_best - lb);

                z_new = obj.mp_solve(z, bundle, level, C);
                step = z_new - z;
                z = z_new;

                [f_new, g_new] = obj.svr_dual_function(z_new, K, y, epsilon);

                f_values(iter) = f_new;
                f_times(iter) = toc;
                % addpoints(h, iter, level);
                % drawnow;
                fprintf('Iter: %d | f(x): %.6f | Grad norm: %.6e | Step norm: %.6e\n', ...
                    iter, f_new, norm(g_new), norm(step));
                
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

                if norm(step) < obj.tol
                    break;
                end

            end

            alpha = z;
        end
    end

    methods (Access=private)
        
        function [f, g] = svr_dual_function(~, x, K, y, epsilon)
            f = 0.5 * x' * (K * x) + epsilon * sum(abs(x)) - y' * x;

            g = K * x + epsilon * sign(x) - y;
        end

        function lb = compute_lower_bound(~, bundle, C)
            [n, m] = size(bundle.z);

            A = [bundle.g', -ones(m, 1)];
            b = sum(bundle.g .* bundle.z, 1)' -bundle.f';

            f   = [zeros(n, 1); 1];
            Aeq = [ones(1, n), 0];
            beq = 0;
            lb  = [-C * ones(n, 1); -inf];
            ub  = [ C * ones(n, 1);  inf];

            options = optimoptions('linprog', 'Display', 'off');
            [~, lb, exitflag] = linprog(f, A, b, Aeq, beq, lb, ub, options);

            if exitflag <= 0
                warning('Error while computing lower bound');
                lb = min(bundle.f);
            end
        end

        
        function alpha_opt = mp_solve(~, alpha_hat, bundle, f_level, C)
            n = length(alpha_hat);
            m = length(bundle.f);

            H = blkdiag(speye(n), 0);
            f = sparse([-alpha_hat; 0]);

            A = sparse([bundle.g' -ones(m, 1)]);
            b = sparse(sum(bundle.g .* bundle.z, 1)' - bundle.f');

            Aeq = sparse([ones(1, n) 0]);
            beq = 0;

            lb = sparse([-C * ones(n, 1); -inf]);
            ub = sparse([C * ones(n, 1); f_level]);

            options = optimoptions('quadprog', 'Display', 'off');

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