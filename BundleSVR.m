classdef BundleSVR < SVR
    methods
        function fit_bundle(obj, X, Y)
            
            obj.X_train = X;
            n = length(Y);
            K = obj.kernel_function.compute(X, X);
            
            f_dual = @(z) 0.5*(z(1:n)-z(n+1:end))' * K * (z(1:n)-z(n+1:end)) + ...
                obj.epsilon * sum(z(1:n) + z(n+1:end)) - Y'*(z(1:n)-z(n+1:end));
            
            grad_dual = @(z) [ K*(z(1:n)-z(n+1:end)) + obj.epsilon - Y;
                              -K*(z(1:n)-z(n+1:end)) + obj.epsilon + Y ];
            
            % Params
            max_iter = 100;
            tol = 1e-4;
            z_current = zeros(2*n, 1);  % Punto iniziale: vettore [alpha; alpha*] tutti a zero
            bundle_z = [];   % Memorizza i punti valutati
            bundle_f = [];   % Memorizza i valori della funzione
            bundle_g = [];   % Memorizza i sottogradienti
            f_vals = [];     % Per monitorare la convergenza
            
            % Vincolo di uguaglianza: sum(alpha - alpha*) = 0
            Aeq = [ones(1, n), -ones(1, n)];
            beq = 0;
            lb = zeros(2*n, 1);
            ub = obj.C * ones(2*n, 1);
            
            for iter = 1:max_iter
                f_val = f_dual(z_current);
                g_val = grad_dual(z_current);
                
                % Aggiungi il punto corrente al bundle
                bundle_z = [bundle_z, z_current];
                bundle_f = [bundle_f; f_val];
                bundle_g = [bundle_g, g_val];
                f_vals(end+1) = f_val;
                
                % Master Problem:
                % Variabili: z (dimensione 2*n) e t (scalare) → totale 2*n+1 variabili.
                % Per ogni punto del bundle imponiamo:
                %    t >= f(z_i) + g_i'*(z - z_i)
                m = size(bundle_z, 2);
                A_bundle = zeros(m, 2*n+1);
                b_bundle = zeros(m, 1);
                for i = 1:m
                    A_bundle(i, 1:2*n) = -bundle_g(:, i)';
                    A_bundle(i, end) = 1;
                    b_bundle(i) = -bundle_f(i) + bundle_g(:, i)' * bundle_z(:, i);
                end
                
                % Vincolo di uguaglianza esteso: Aeq * z + 0*t = beq
                Aeq_full = [Aeq, 0];
                lb_full = [lb; -inf];  % t è non vincolata inferiormente
                ub_full = [ub; inf];
                
                % minimize t
                f_master = [zeros(2*n, 1); 1];
                options_linprog = optimoptions('linprog', 'Display', 'off');
                [sol, ~, exitflag] = linprog(f_master, A_bundle, b_bundle, Aeq_full, beq, lb_full, ub_full, options_linprog);
                
                if exitflag ~= 1
                    warning('Il Master Problem non converge alla iterazione %d.', iter);
                    break;
                end

                z_new = sol(1:2*n);
                
                % Controlla la convergenza: se la variazione della funzione è piccola, esci
                if abs(f_dual(z_new) - f_val) < tol
                    fprintf('Convergenza raggiunta alla iterazione %d\n', iter);
                    z_current = z_new;
                    f_vals(end+1) = f_dual(z_current);
                    break;
                end
                
                z_current = z_new;
            end
            
            alpha_pos = z_current(1:n);
            alpha_neg = z_current(n+1:end);
            obj.alpha_svr = alpha_pos - alpha_neg;
            obj.bias = mean(Y - K * obj.alpha_svr);
            
            % Plot della convergenza del bundle method (valore della funzione duale vs. iterazioni)
            figure;
            plot(f_vals, '-o', 'LineWidth', 2);
            xlabel('Iterazione');
            ylabel('Valore funzione obiettivo duale');
            title('Convergenza del Bundle Method');
            grid on;
        end
    end
end
