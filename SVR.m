classdef SVR < handle
    properties(Access=protected)
        kernel_function
        C double
        epsilon double
        alpha_svr
        bias
        X_train
        opt
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
                z = obj.opt.optimize(K, Y, obj.C,obj.epsilon);
            else
                H = [K, -K; -K, K];
                f = [obj.epsilon + Y; obj.epsilon - Y]';
                Aeq = [ones(1, length(Y)), -ones(1, length(Y))];
                beq = 0;
                lb = zeros(2*n, 1);
                ub = obj.C * ones(2*n, 1);

                if exist('mosekopt', 'file') == 3
                    options = mskoptimset('Display', 'off');
                else
                    options = optimoptions('quadprog', 'Display', 'off');
                end
                z = quadprog(H, -f, [], [], Aeq, beq, lb, ub, [], options);
            end
            
            obj.alpha_svr = z;
            % Identify support vectors
            sv_indices = find(abs(obj.alpha_svr) > obj.tol);
            
            if isempty(sv_indices)
                X_sv = [];
                Y_sv = [];
                warning("Support vectors not found");

                obj.bias = mean(Y - (K * obj.alpha_svr));
            else
                X_sv = obj.X_train(sv_indices, :);
                Y_sv = Y(sv_indices);

                disp("alpha max: " + max(obj.alpha_svr));
                disp("alpha min: " + min(obj.alpha_svr));

                obj.bias = mean(Y(sv_indices) - K(sv_indices, :) * obj.alpha_svr);
            end
        end

        function y_pred = predict(obj, X)
            K_test = obj.kernel_function.compute(X, obj.X_train);
            y_pred = K_test * obj.alpha_svr + obj.bias;
        end
    end
end