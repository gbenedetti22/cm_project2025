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
            z0 = zeros(2*n, 1);
            z = obj.project(z0, C);
            [f, g] = obj.svr_dual_function(z, K, y, obj.epsilon);
            f_best = f;

            bundle.z = z;
            bundle.f = f;
            bundle.g = g;

            t = obj.tol;
            
            loading_bar = waitbar(0,'Computing LBM...');
            for iter = 1:obj.max_iter
                waitbar(iter/obj.max_iter, loading_bar, "Computing LBM (" + (iter) + "/" + obj.max_iter + ")");

                level = obj.theta * f + (1 - obj.theta) * f_best;

                z_new = obj.mp_solve(z, bundle, t, C);

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
    end

    methods (Access=private)
        function [f, g] = svr_dual_function(~, z, K, y, epsilon)
            n = length(y);
            alpha = z(1:n);
            alphaStar = z(n+1:end);
            diffAlpha = alpha - alphaStar;

            f = 0.5 * diffAlpha' * (K * diffAlpha) + epsilon * sum(alpha + alphaStar) - y' * diffAlpha;

            Kdiff = K * diffAlpha;
            g = [Kdiff + epsilon - y; -Kdiff + epsilon + y];
        end

        function z_proj = project(~, z, C)
            n = numel(z)/2;

            alpha = min(max(z(1:n), 0), C);
            alphaStar = min(max(z(n+1:end), 0), C);

            diff = (sum(alpha) - sum(alphaStar)) / (2*n);
            alpha = min(max(alpha - diff, 0), C);
            alphaStar = min(max(alphaStar + diff, 0), C);

            z_proj = [alpha; alphaStar];
        end


        function z_opt = mp_solve(~, z_current, bundle, t, C)
            n2 = length(z_current);
            n = n2/2;
            m = length(bundle.f);

            H = sparse(1:n2, 1:n2, 1/t, n2, n2);
            H = blkdiag(H, 0);

            f = sparse([-(1/t) * z_current; 1]);
            
            A_bundle = sparse([bundle.g' -ones(m,1)]);
            b_bundle = sparse(sum(bundle.g .* bundle.z, 1)' - bundle.f');

            Aeq = sparse([ones(1,n), -ones(1,n), 0]);
            beq = 0;

            lb = sparse([zeros(n2,1); -inf]);
            ub = sparse([C * ones(n2,1);  inf]);

            if exist('mosekopt', 'file') == 3
                options = mskoptimset('Display', 'off');
            else
                options = optimoptions('Display', 'off', 'Algorithm', 'interior-point-convex');
            end
            
            [z_sol, ~, exitflag] = quadprog(H, f, A_bundle, b_bundle, Aeq, beq, lb, ub, [], options);

            if exitflag <= 0
                warning('quadprog non ha trovato una soluzione valida. Manteniamo la soluzione corrente.');
                z_opt = z_current;
            else
                z_opt = z_sol(1:n2);
            end
        end

    end
end