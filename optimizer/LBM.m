classdef LBM < handle    
    properties(Access=private)
        max_iter
        epsilon
        tol
        theta
        max_constraints
        qp_ratio    %now: Numero di iterazioni per ciascuna modalità
        % before: 0 -> sempre subgradiente, 100 -> sempre quadprog, altrimenti modalità "auto"
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
            % loading_bar = waitbar(0, 'Computing LBM...');
            h = animatedline('LineStyle','-', 'Marker','none', 'LineWidth', 2);
            
            % mode_qp = true;
            % iter_count = 0;

            for iter = 1:obj.max_iter
                %waitbar(iter/obj.max_iter, loading_bar, "Computing LBM (" + iter + "/" + obj.max_iter + ")");
                level = obj.theta * f + (1 - obj.theta) * f_best;
                %%%%IMPLEMENTAZIONE CON CAMBIO ex:primi 10 qp- resto subg %%%%
                % if iter <=20
                %     z_new = obj.mp_quadprog_solve(z, bundle, t, C);
                % else
                %     z_new = obj.subgradient_step(z, bundle, t, C, level);
                %%%%IMPLEMENTAZIONE ALTERNATA ex:10-10 %%%%
                % if mode_qp
                %     % Sempre quadprog
                %     z_new = obj.mp_quadprog_solve(z, bundle, t, C);
                % else
                %     z_new = obj.subgradient_step(z, bundle, t, C, level);
                
                %%%%IMPLEMENTAZIONE PERCENTUALE e:10% %%%%
                if obj.qp_ratio == 100
                    % Sempre quadprog
                    z_new = obj.mp_quadprog_solve(z, bundle, t, C);
                elseif obj.qp_ratio == 0
                    % Sempre subgradiente
                    z_new = obj.subgradient_step(z, bundle, t, C, level);
                else
                    % Usa quadprog ogni round(100/qp_ratio) iterazioni
                    if mod(iter, round(100/obj.qp_ratio)) == 0
                        z_new = obj.mp_quadprog_solve(z, bundle, t, C);
                    else
                        z_new = obj.subgradient_step(z, bundle, t, C, level);
                    end
                end

                % iter_count = iter_count + 1;
                % if iter_count >= obj.max_iter
                %     mode_qp = ~mode_qp;
                %     iter_count = 0;
                % end
                step = z_new - z;
                [f_new, g_new] = obj.svr_dual_function(z_new, K, y, obj.epsilon);
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
                
                bundle.z = [bundle.z, z_new];
                bundle.f = [bundle.f, f_new];
                bundle.g = [bundle.g, g_new];
                
                if f_new <= level
                    z = z_new;
                    f = f_new;
                    g = g_new;
                else
                    t = t * 0.5;
                end

                if norm(step) < obj.tol
                    break;
                end
            end

            %delete(loading_bar);
            alpha = z;
        end

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

        function z_opt = mp_quadprog_solve(~, z_current, bundle, t, C)
            % Function to minimize: Q(z,s) = s + 1/2t z-z_current^2
            % That is the objective function of the level bundle method

            n2 = length(z_current);
            n = n2/2;
            m = length(bundle.f);

            % H represent the quadratic term of the objective function, so
            % contains 1/t * I
            H = sparse(1:n2, 1:n2, 1/t, n2, n2);
            H = blkdiag(H, 0);

            % linear term
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
                options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');
            end
            
            [z_sol, ~, exitflag] = quadprog(H, f, A_bundle, b_bundle, Aeq, beq, lb, ub, [], options);

            if exitflag <= 0
                warning('quadprog non ha trovato una soluzione valida. Manteniamo la soluzione corrente.');
                z_opt = z_current;
            else
                z_opt = z_sol(1:n2);
            end
        end

        function z_sg = subgradient_step(obj, z_current, bundle, t, C, level)
            z_diff = z_current - bundle.z;
            lin_approx = bundle.f + sum(bundle.g .* z_diff, 1);
            active_cuts = lin_approx >= level;
            
            if any(active_cuts)
                agg_g = sum(bundle.g(:, active_cuts), 2);

            else
                agg_g = zeros(size(z_current));
            end
            
            if norm(agg_g) < 1e-12
                z_sg = z_current;
                return;
            end
            
            step = t * agg_g;
            z_new = z_current - step;
            z_sg = obj.project(z_new, C);
        end
    end
end
