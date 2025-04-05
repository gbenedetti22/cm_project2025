classdef SVR < handle
    properties(Access=private)
        kernel_function
        C 
        epsilon 
        alpha_svr
        bias
        X_train
        opt
        tol
        f_values
    end

    methods
        function obj = SVR(params)
            required_fields = {'kernel_function', 'C', 'epsilon'};
            for f = required_fields
                if ~isfield(params, f{1})
                    error('Parametro mancante: %s', f{1});
                end
            end
            
            if ~isa(params.kernel_function, 'KernelFunction')
                error('Kernel non valido');
            end
            obj.kernel_function = params.kernel_function;
            obj.C = params.C;
            obj.epsilon = params.epsilon;
            
            if isfield(params, 'opt')
                obj.opt = params.opt;
            else
                obj.opt = [];
            end
            
            if isfield(params, 'tol')
                obj.tol = params.tol;
            else
                obj.tol = 1e-5; % Valore di default
            end
        end

        function [x, f_values] = fit(obj, X, Y)
            obj.X_train = X;
            K = obj.kernel_function.compute(X, X);
            % Aggiungi regolarizzazione numerica
            K = K + 1e-6 * eye(size(K));
            
            if isa(obj.opt, 'LBM')
                [x, f_values] = obj.opt.optimize(K, Y, obj.C, obj.epsilon);
            else
                n = length(Y);
                svr_dual = @(x) obj.svr_dual_function(x, K, Y, obj.epsilon);
                alpha0 = zeros(n, 1);
                
                % Inizializza f_values e configura output function
                obj.f_values = [];
                outHandle  = @(x, optimValues, state) obj.outfun(x, optimValues, state);
                
                 options = optimoptions('fmincon', 'Display', 'iter', ...
                    'SpecifyObjectiveGradient', true, 'MaxIterations', 50, ...
                    'OutputFcn', outHandle);

                Aeq = ones(1, n);
                beq = 0;

                A = [];
                b = [];

                lb = -obj.C * ones(n, 1);
                ub = obj.C * ones(n, 1);

                x = fmincon(svr_dual, alpha0, A, b, Aeq, beq, lb, ub, [], options);
                
                
                f_values = obj.f_values; % Recupera i valori registrati
            end
            
            obj.alpha_svr = x;
            sv_indices = abs(obj.alpha_svr) > obj.tol;
            
            if any(sv_indices)
                obj.bias = mean(Y(sv_indices) - K(sv_indices, :) * x);
            else
                obj.bias = mean(Y - K * x);
            end
        end

        function [f, g] = svr_dual_function(~, x, K, y, epsilon)
            f = 0.5 * x' * (K * x) + epsilon * sum(abs(x)) - y' * x;
            g = K * x + epsilon * sign(x) - y;
        end

        function y_pred = predict(obj, X)
            K_test = obj.kernel_function.compute(X, obj.X_train);
            y_pred = K_test * obj.alpha_svr + obj.bias;
        end

        % Output function per registrare i valori della funzione obiettivo
        function stop = outfun(obj, ~, optimValues, state)
            stop = false;
            if strcmp(state, 'iter')
                obj.f_values = [obj.f_values; optimValues.fval];
            end
        end
    end
end