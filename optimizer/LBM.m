classdef LBM < handle    
    properties(Access=private)
        max_iter
        epsilon
        tol
        theta
        max_constraints
    end
    
    methods
        function obj = LBM(params)
            if ~isstruct(params)
                error('Input must be a struct');
            end
            obj = init_parameters(obj, params);
        end
        
        function obj = init_parameters(obj, params)
            % Valori di default
            default_params = struct(...
                'max_iter', 500,...
                'epsilon', 1e-6,...
                'tol', 1e-6,...
                'theta', 0.5,...
                'max_constraints', 1000....
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
            
            validateattributes(obj.max_iter, {'numeric'}, {'positive', 'integer'});
            validateattributes(obj.epsilon, {'numeric'}, {'positive'});
        end
        
        function alpha = optimize(obj, K, y, C, epsilon)
            n = length(y);
            z = zeros(n, 1);
            alpha= z;
            %alpha = obj.project(z, C);
            [f, g] = obj.svr_dual_function(alpha, K, y, epsilon);
            
            f_best = f;

            bundle.z = z;
            bundle.f = f;
            bundle.g = g;
            t = obj.tol;

            % loading_bar = waitbar(0, 'Computing LBM...');

            h = animatedline('LineStyle','-', 'Marker','none', 'LineWidth', 2);
            for iter = 1:obj.max_iter

                level = obj.theta * f + (1 - obj.theta) * f_best;
                z_new = obj.mp_solve(z, bundle, level, C);
                step = z_new - z;
                [f_new, g_new] = obj.svr_dual_function(z_new, K, y, epsilon);
                addpoints(h, iter, f_new);
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

                if norm(step) < t
                    break;
                end
            end
            alpha = z;
        end

        function [f, g] = svr_dual_function(~, gamma, K, y, epsilon)
            f = 0.5 * gamma' * (K * gamma) + epsilon * sum(abs(gamma)) - y' * gamma;
            g_smooth = K * gamma - y;
            % Subgradient for L1 term
            g_l1 = epsilon * sign(gamma);
            % Handle gamma_i = 0 by choosing subgradient in [-epsilon, epsilon]
            zero_indices = (gamma == 0);
            g_l1(zero_indices) = epsilon * sign(g_smooth(zero_indices));
            g_l1(zero_indices & (g_smooth == 0)) = 0; % If smooth grad is zero, subgradient can be 0
            g = g_smooth + g_l1;
        end
        

        function gamma_proj = project(~, gamma, C)
            gamma = max(min(gamma, C), -C);  % Initial clipping
            for k = 1:5
                % Sum correction
                avg = mean(gamma);
                gamma = gamma - avg;
                % Re-clipping
                gamma = max(min(gamma, C), -C);
            end
            gamma_proj = gamma;
        end


        function alpha_opt = mp_solve(~, alpha_current, bundle, f_level, C)
            n = length(alpha_current);
            m = length(bundle.f);
            H = blkdiag(speye(n), 0);
            
            f = sparse([-alpha_current; 0]);
            
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
            sol = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], options);
            
            if isempty(sol)
                alpha_opt = alpha_current;
            else
                alpha_opt = sol(1:n);
            end
        end
    end
end