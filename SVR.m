classdef SVR < handle
    properties(Access=private)
        kernel_function
        C double
        epsilon double
        alpha_svr
        bias
        X_train
        opt
        mu_X
        sigma_X
        mu_Y
        sigma_Y
    end

    methods
        function obj = SVR(kernel_function, C, epsilon, opt)
           if isa(kernel_function, 'KernelFunction')
                obj.kernel_function = kernel_function;
                obj.C = C;
                obj.epsilon = epsilon;

                if nargin < 4
                    opt = false;
                else
                    if not(isobject(opt)) || not(isa(opt, 'LBM'))
                        error('invalid optimizer');
                    end
                end

                obj.opt = opt;
            else
                error('kernel_function must be a subclass of KernelFunction');
            end

        end

        function fit(obj, X, Y)
            obj.X_train = X;
            [obj.X_train, obj.mu_X, obj.sigma_X] = zscore(X); 
            [Y, obj.mu_Y, obj.sigma_Y] = zscore(Y);
            obj.X_train(:, obj.sigma_X == 0) = 0;
            K = obj.kernel_function.compute(X, X);
            K = K + 1e-8 * eye(size(K));

            H = [K, -K; -K, K];
            f = [obj.epsilon + Y; obj.epsilon - Y]';
            Aeq = [ones(1, length(Y)), -ones(1, length(Y))];
            beq = 0;
            
            lb = zeros(2*length(Y), 1);
            ub = obj.C * ones(2*length(Y), 1);
            
            if isa(obj.opt, 'LBM')
                alpha = obj.opt.optimize(Y, H, f(:));
            else
                options = optimoptions('quadprog', 'Display', 'off');
                alpha = quadprog(H, -f, [], [], Aeq, beq, lb, ub, [], options);
            end
            
            alpha_pos = alpha(1:length(Y));
            alpha_neg = alpha(length(Y)+1:end);
            obj.alpha_svr = alpha_pos - alpha_neg;
            
            if isa(obj.opt, 'LBM')
                support_indices = find(abs(obj.alpha_svr) > 1e-5 & abs(obj.alpha_svr) < obj.C);
                
                if isempty(support_indices)
                    obj.bias = mean(Y - K * obj.alpha_svr);  
                else
                    obj.bias = mean(Y(support_indices) - K(support_indices, :) * obj.alpha_svr);
                end
            else
                obj.bias = mean(Y - (K * obj.alpha_svr));
            end
        end

        function y_pred = predict(obj, X)
            % Normalizza X usando mu e sigma di training
            X_normalized = (X - obj.mu_X) ./ obj.sigma_X;
            X_normalized(:, obj.sigma_X == 0) = 0;  % Gestisci colonne costanti

            % Calcola kernel e predici
            K_test = obj.kernel_function.compute(X_normalized, obj.X_train);
            y_pred_normalized = K_test * obj.alpha_svr + obj.bias;

            % Denormalizza le predizioni (se Y è stato normalizzato)
            y_pred = y_pred_normalized * obj.sigma_Y + obj.mu_Y;
        end
    end
end