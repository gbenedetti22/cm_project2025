classdef SVR_norm_kTrick < handle
    properties(Access=protected)
        kernel_function
        C double
        epsilon double
        alpha_svr
        bias
        X_train
        opt
        tol
        % Proprietà per la normalizzazione
        mu
        sigma
    end

    methods
        function obj = SVR_norm_kTrick(kernel_function, C, epsilon, opt, tol)
            if isa(kernel_function, 'KernelFunction') || isa(kernel_function, 'RBFKernel_wTrick')
                obj.kernel_function = kernel_function;
                obj.C = C;
                obj.epsilon = epsilon;

                if nargin < 4
                    opt = false;
                else
                    if ~isobject(opt) || ~isa(opt, 'LBM')
                        error('invalid optimizer');
                    end
                end

                if nargin < 5
                    tol = 1e-5;
                end

                obj.opt = opt;
                obj.tol = tol;
            else
                error('kernel_function must be a subclass of KernelFunction or RBFKernel');
            end
        end

        function [X_sv, Y_sv] = fit(obj, X, Y)
            % Normalizzazione: calcola media e std e applica scaling
            obj.mu = mean(X, 1);
            obj.sigma = std(X, 1);
            % Evita divisioni per zero
            obj.sigma(obj.sigma == 0) = 1;
            X_norm = (X - obj.mu) ./ obj.sigma;
            
            obj.X_train = X_norm;
            % Applica il kernel trick: calcola la matrice kernel sui dati normalizzati
            K = obj.kernel_function.compute(X_norm, X_norm);

            if isa(obj.opt, 'LBM')
                z = obj.opt.optimize(K, Y, obj.C);
            else
                H = [K, -K; -K, K];
                f = [obj.epsilon + Y; obj.epsilon - Y]';
                Aeq = [ones(1, length(Y)), -ones(1, length(Y))];
                beq = 0;
                lb = zeros(2*length(Y), 1);
                ub = obj.C * ones(2*length(Y), 1);
                options = optimoptions('quadprog', 'Display', 'off');
                z = quadprog(H, -f, [], [], Aeq, beq, lb, ub, [], options);
            end
            
            alpha_pos = z(1:length(Y));
            alpha_neg = z(length(Y)+1:end);
            obj.alpha_svr = alpha_pos - alpha_neg;
            
            sv_indices = find(abs(obj.alpha_svr) > obj.tol);
            
            if isempty(sv_indices)
                X_sv = [];
                Y_sv = [];
                warning("Support vectors not found");
                obj.bias = mean(Y - (K * obj.alpha_svr));
            else
                X_sv = obj.X_train(sv_indices, :);
                Y_sv = Y(sv_indices);
                obj.bias = mean(Y(sv_indices) - K(sv_indices, :) * obj.alpha_svr);
            end
        end

        function y_pred = predict(obj, X)
            % Normalizza i dati di test utilizzando i parametri calcolati in training
            X_norm = (X - obj.mu) ./ obj.sigma;
            % Applica il kernel trick per la predizione
            K_test = obj.kernel_function.compute(X_norm, obj.X_train);
            y_pred = K_test * obj.alpha_svr + obj.bias;
        end
    end
end
