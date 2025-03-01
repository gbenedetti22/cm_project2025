classdef SVR < handle
    properties(Access=private)
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
        function obj = SVR(kernel_function, C, epsilon, opt, tol)
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
                   
                if nargin < 5
                    tol = 1e-5;
                end

                obj.opt = opt;
                obj.tol = tol;
            else
                error('kernel_function must be a subclass of KernelFunction');
            end

        end

        function [X_sv, Y_sv] = fit(obj, X, Y)
            obj.X_train = X;
            K = obj.kernel_function.compute(X, X);

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
            K_test = obj.kernel_function.compute(X, obj.X_train);
            y_pred = K_test * obj.alpha_svr + obj.bias;
        end
    end
end