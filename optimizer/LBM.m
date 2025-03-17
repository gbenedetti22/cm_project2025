classdef LBM < handle    
    properties(Access=private)
        max_iter
        epsilon
        tol
        theta
        max_constraints
        qp_ratio   % 0 -> sempre subgradiente, 100 -> sempre quadprog, altrimenti modalità "auto"
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
                'max_constraints', 1000,...
                'qp_ratio', 10....
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
        
        function gamma = optimize(obj, K, y, C, epsilon)
            n = length(y);
            gamma0 = zeros(n, 1);
            gamma = obj.project(gamma0, C);
            [f, g] = obj.svr_dual_function(gamma, K, y, epsilon);
            f_best = f;

            bundle.z = gamma;
            bundle.f = f;
            bundle.g = g;

            t = obj.tol;
            h = animatedline('LineStyle','-', 'Marker','none', 'LineWidth', 2);
            
            for iter = 1:obj.max_iter
                level = obj.theta * f + (1 - obj.theta) * f_best;
                
                if obj.qp_ratio == 100
                    gamma_new = obj.mp_quadprog_solve(gamma, bundle, t, C);
                elseif obj.qp_ratio == 0
                    gamma_new = obj.subgradient_step(gamma, bundle, t, C, level);
                else
                    if mod(iter, round(100/obj.qp_ratio)) == 0
                        gamma_new = obj.mp_quadprog_solve(gamma, bundle, t, C);
                    else
                        gamma_new = obj.subgradient_step(gamma, bundle, t, C, level);
                    end
                end

                step = gamma_new - gamma;
                [f_new, g_new] = obj.svr_dual_function(gamma_new, K, y, epsilon);
                addpoints(h, iter, f_new);
                drawnow;
                
                if f_new < f_best
                    f_best = f_new;
                end
                
                if size(bundle.z, 2) >= obj.max_constraints
                    bundle.z = bundle.z(:, 2:end);
                    bundle.f = bundle.f(2:end);
                    bundle.g = bundle.g(:, 2:end);
                end
                
                bundle.z = [bundle.z, gamma_new];
                bundle.f = [bundle.f, f_new];
                bundle.g = [bundle.g, g_new];
                
                if f_new <= level
                    gamma = gamma_new;
                    f = f_new;
                    g = g_new;
                else
                    t = t * 0.5;
                end

                if norm(step) < obj.tol
                    break;
                end
            end
        end

        function [f, g] = svr_dual_function(~, gamma, K, y, epsilon)
            f = 0.5 * gamma' * K * gamma + epsilon * sum(abs(gamma)) - y' * gamma;
            g_smooth = K * gamma - y;
            % Subgradient for L1 term
            g_l1 = epsilon * sign(gamma);
            % Handle gamma_i = 0 by choosing subgradient in [-epsilon, epsilon]
            zero_indices = (gamma == 0);
            g_l1(zero_indices) = epsilon * sign(g_smooth(zero_indices));
            g_l1(zero_indices & (g_smooth == 0)) = 0; % If smooth grad is zero, subgradient can be 0
            g = g_smooth + g_l1;
        end
        
        %precise method (slow)
        % function gamma_proj = project(~, gamma, C)
        %     n = length(gamma);
        %     H = eye(n);
        %     f = -gamma;
        %     Aeq = ones(1, n);
        %     beq = 0;
        %     lb = -C * ones(n, 1);
        %     ub = C * ones(n, 1);
        %     options = optimoptions('quadprog', 'Display', 'off');
        %     gamma_proj = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], options);
        % end

        %iterative approach (faster but less precise)
        function gamma_proj = project(~, gamma, C)
            gamma = max(min(gamma, C), -C);  % Clipping iniziale
            for k = 1:5
                % Correzione della somma
                avg = mean(gamma);
                gamma = gamma - avg;
                % Ri-clipping
                gamma = max(min(gamma, C), -C);
            end
            gamma_proj = gamma;
        end


        function gamma_opt = mp_quadprog_solve(~, gamma_current, bundle, t, C)
            n = length(gamma_current);
            m = length(bundle.f);

            H = sparse(1:n, 1:n, 1/t, n, n);
            H = blkdiag(H, 0);
            
            f = [-(1/t) * gamma_current; 1];
            
            A_bundle = [bundle.g' -ones(m,1)];
            b_bundle = sum(bundle.g .* bundle.z, 1)' - bundle.f';
            
            Aeq = sparse([ones(1, n) 0]);
            beq = 0;
            
            lb = [-C * ones(n,1); -inf];
            ub = [C * ones(n,1); inf];
            
            options = optimoptions('quadprog', 'Display', 'off');
            sol = quadprog(H, f, A_bundle, b_bundle, Aeq, beq, lb, ub, [], options);
            
            if isempty(sol)
                gamma_opt = gamma_current;
            else
                gamma_opt = sol(1:n);
                gamma_opt = obj.project(gamma_opt, C); % Chiamata corretta
            end
        end

        function gamma_sg = subgradient_step(obj, gamma_current, bundle, t, C, level)
            gamma_diff = gamma_current - bundle.z;
            lin_approx = bundle.f + sum(bundle.g .* gamma_diff, 1);
            active_cuts = lin_approx >= level;
            
            if any(active_cuts)
                agg_g = mean(bundle.g(:, active_cuts), 2);
            else
                agg_g = zeros(size(gamma_current));
            end
            
            gamma_new = gamma_current - t * agg_g;
            gamma_sg = obj.project(gamma_new, C);
        end
    end
end