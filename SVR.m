classdef SVR < handle
    properties(Access=protected)
        kernel_function
        C double
        epsilon double
        alpha_svr   % In questa formulazione compatta, alpha_svr coincide con w = alpha - alpha*
        bias
        X_train
        opt         % Se è un oggetto LBM, viene usato il Level Bundle Method
        tol
    end

    methods
        function obj = SVR(params)
            % Verifica campi obbligatori
            required_fields = {'kernel_function', 'C', 'epsilon'};
            for f = required_fields
                if ~isfield(params, f{1})
                    error('Struct params must contain field: %s', f{1});
                end
            end

            % Validazione kernel
            if ~isa(params.kernel_function, 'KernelFunction')
                error('kernel_function must be a subclass of KernelFunction');
            end
            obj.kernel_function = params.kernel_function;

            % Assegna parametri obbligatori
            obj.C = params.C;
            obj.epsilon = params.epsilon;

            % Assegna parametri opzionali con default
            if isfield(params, 'opt')
                if ~isempty(params.opt) && ( ~isobject(params.opt) || ~isa(params.opt, 'LBM') )
                    error('opt must be a LBM object or empty');
                end
                obj.opt = params.opt;
            else
                obj.opt = false;
            end

            if isfield(params, 'tol')
                obj.tol = params.tol;
            else
                obj.tol = 1e-5;
            end
        end

        function [X_sv, Y_sv] = fit(obj, X, Y)
            obj.X_train = X;
            K = obj.kernel_function.compute(X, X);

            if isa(obj.opt, 'LBM')
                % Nel caso LBM, utilizziamo la formulazione compatta:
                % La variabile di ottimizzazione è z = [w; u]
                % w = alpha - alpha* e u viene usato per gestire il termine epsilon*(alpha+alpha*) 
                % (tramite il vincolo u >= |w|)
                [w, ~] = obj.opt.optimize(K, Y, obj.C);
                z = [w; zeros(length(w),1)]; % u non viene utilizzato per le previsioni
            else
                % Formulazione classica: problema duale con variabili separate
                H = [K, -K; -K, K];
                f = [obj.epsilon + Y; obj.epsilon - Y]';
                Aeq = [ones(1, length(Y)), -ones(1, length(Y))];
                beq = 0;
                lb = zeros(2*length(Y), 1);
                ub = obj.C * ones(2*length(Y), 1);
                options = optimoptions('quadprog', 'Display', 'off');
                z = quadprog(H, -f, [], [], Aeq, beq, lb, ub, [], options);
                w = z(1:length(Y)) - z(length(Y)+1:end);  % w = alpha - alpha*
            end

            obj.alpha_svr = w;  % In entrambe le formulazioni, la componente w viene usata per la predizione

            % Identifica i support vector usando una soglia di tolleranza
            sv_indices = find(abs(obj.alpha_svr) > obj.tol);
            if isempty(sv_indices)
                X_sv = [];
                Y_sv = [];
                warning("Support vectors not found");
                obj.bias = mean(Y - K * obj.alpha_svr);
            else
                X_sv = obj.X_train(sv_indices, :);
                Y_sv = Y(sv_indices);
                obj.bias = mean(Y(sv_indices) - K(sv_indices, :) * obj.alpha_svr);
            end
        end

        function y_pred = predict(obj, X)
            K_test = obj.kernel_function.compute(X, obj.X_train);
            y_pred = K_test * obj.alpha_svr + obj.bias;
        end
    end
end
