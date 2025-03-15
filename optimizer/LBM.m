classdef LBM < handle    
    properties(Access=private)
        max_iter
        epsilon
        tol
        theta
        max_constraints
        qp_ratio    % 0: sempre subgradiente, 100: sempre quadprog, altrimenti modalità ibrida
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
                'max_iter', 500, ...
                'epsilon', 1e-6, ...
                'tol', 1e-6, ...
                'theta', 0.5, ...
                'max_constraints', 1000, ...
                'qp_ratio', 10 ...
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
        
        % La funzione optimize restituisce le variabili compattate:
        % w = alpha - alpha* e u, da cui ricostruire il modello SVR
        function [w, u] = optimize(obj, K, y, C)
            n = length(y);
            % Inizialmente, poniamo w = 0 e u = C (scelta ammissibile: C >= |0|)
            z0 = [zeros(n,1); C * ones(n,1)];
            z = obj.project(z0, C);
            [f, g] = obj.svr_dual_function(z, K, y, obj.epsilon);
            f_best = f;

            % Inizializza il bundle con il punto corrente
            bundle.z = z;
            bundle.f = f;
            bundle.g = g;

            t = obj.tol;
            h = animatedline('LineStyle','-', 'Marker','none', 'LineWidth', 2);
            
            for iter = 1:obj.max_iter
                level = obj.theta * f + (1 - obj.theta) * f_best;
                
                % Alterna la modalità in base a qp_ratio
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

                step = z_new - z;
                [f_new, g_new] = obj.svr_dual_function(z_new, K, y, obj.epsilon);
                addpoints(h, iter, f_new);
                drawnow;
                
                if f_new < f_best
                    f_best = f_new;
                end
                
                % Troncamento del bundle se supera il numero massimo di vincoli
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

            % Estrae le componenti: le prime n componenti sono w, le restanti u
            w = z(1:n);
            u = z(n+1:end);
        end

        % Funzione duale compatta per SVR
        % Input: z = [w; u], con w in R^n e u in R^n
        % f(w,u) = 0.5 * w' * K * w + epsilon * sum(u) - y' * w
        function [f, g] = svr_dual_function(~, z, K, y, epsilon)
            n = length(y);
            w = z(1:n);
            u = z(n+1:end);
            
            f = 0.5 * w' * K * w + epsilon * sum(u) - y' * w;
            
            % Calcolo dei subgradienti:
            % Per w: grad_w = K*w - y
            % Per u: grad_u = epsilon (visto che la derivata di epsilon*u è epsilon)
            g = [K * w - y; epsilon * ones(n, 1)];
            
            % Nota: I termini derivanti dal vincolo u >= |w| vengono gestiti tramite le cutting planes (bundle)
        end

        % Funzione di proiezione per assicurare la fattibilità:
        % - Si impone che la media di w sia zero (equivalente al vincolo sum(w)=0)
        % - Si impone che per ogni i, u_i >= |w_i| e 0 <= u_i <= 2C
        function z_proj = project(~, z, C)
            n = length(z)/2;
            w = z(1:n);
            u = z(n+1:end);
            
            % Forza il vincolo di equilibrio: sum(w) = 0
            w = w - mean(w);
            
            % Per ogni componente, impone u_i >= |w_i| e rispetta i limiti [0, 2C]
            u = max(u, abs(w));
            u = min(max(u, 0), 2 * C);
            
            z_proj = [w; u];
        end

        % Risolutore del problema master via quadprog
        % Si risolve:
        %   min (1/(2t))||z - z_current||^2 + s
        % soggetto a:
        %   cutting plane constraints: bundle.f + bundle.g'*(z - bundle.z) <= s
        %   vincoli: sum(w)=0, 0 <= u <= 2C (soglia su z gestita via lb/ub)
        function z_opt = mp_quadprog_solve(~, z_current, bundle, t, C)
            n2 = length(z_current);
            n = n2/2;
            m = length(bundle.f);

            % Matrice quadratica: H = (1/t)*I per z (esteso a dimensione n2) e zero per s
            H = sparse(1:n2, 1:n2, 1/t, n2, n2);
            H = blkdiag(H, 0);

            % Termine lineare: f_lin = - (1/t)*z_current concatenato con 1 per s
            f_lin = sparse([-(1/t) * z_current; 1]);

            % Vincoli derivanti dal bundle: A_bundle * [z; s] <= b_bundle
            A_bundle = sparse([bundle.g' -ones(m, 1)]);
            b_bundle = sparse(sum(bundle.g .* bundle.z, 1)' - bundle.f');

            % Vincolo di uguaglianza per la condizione sum(w)=0.
            % Le prime n componenti di z sono w.
            Aeq = sparse([ones(1, n), zeros(1, n+1)]);
            beq = 0;

            % Limiti inferiori e superiori:
            % Per w: nessun limite (inf), per u: [0, 2C]
            lb = [-inf(n2, 1); -inf];
            lb(n+1:n2) = 0;
            ub = [inf(n2, 1); inf];
            ub(n+1:n2) = 2 * C;

            options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');
            [z_sol, ~, exitflag] = quadprog(H, f_lin, A_bundle, b_bundle, Aeq, beq, lb, ub, [], options);

            if exitflag <= 0
                warning('quadprog non ha trovato una soluzione valida. Manteniamo la soluzione corrente.');
                z_opt = z_current;
            else
                z_opt = z_sol(1:n2);
            end
        end

        % Aggiornamento via subgradiente aggregato
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
