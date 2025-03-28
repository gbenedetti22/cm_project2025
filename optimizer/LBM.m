classdef LBM < handle    
    properties(Access=private)
        max_iter
        epsilon
        tol
        theta
        max_constraints
    end
    
    methods
        function obj = LBM(max_iter, epsilon, tol, theta, max_constraints)
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

            obj.max_iter = max_iter;
            obj.epsilon = epsilon;
            obj.tol = tol;
            obj.theta = theta;
            obj.max_constraints = max_constraints;
        end
        
        function alpha = optimize(obj, K, y, C)
            n = length(y);
            z = zeros(n, 1);
            [f, g] = obj.svr_dual_function(z, K, y, obj.epsilon);
            f_best = f;

            bundle.z = z;
            bundle.f = f;
            bundle.g = g;
            
            h = animatedline('LineStyle','-', 'Marker','none', 'LineWidth', 2);
            for iter = 1:obj.max_iter
                level = obj.theta * f + (1 - obj.theta) * f_best;

                z_new = obj.mp_solve(z, bundle, level, C);

                step = z_new - z;

                [f_new, g_new] = obj.svr_dual_function(z_new, K, y, obj.epsilon);
                addpoints(h, iter, f_new);
                % for gj = 1:length(g_new)
                %     addpoints(h, iter, g_new(gj));
                % end
                drawnow;
                
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
                end

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

            if exist('mosekopt','file') == 3
                options = mskoptimset('Display','off');
            else
                options = optimoptions('quadprog','Display','off','Algorithm','interior-point-convex');
            end

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