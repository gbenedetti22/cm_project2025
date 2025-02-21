classdef SVR < handle
    properties(Access=private)
        kernel_function
        C double
        epsilon double
        alpha_svr
        bias
        X_train
    end

    methods
        function obj = SVR(kernel_function, C, epsilon)
           if isa(kernel_function, 'KernelFunction')
                obj.kernel_function = kernel_function;
                obj.C = C;
                obj.epsilon = epsilon;
            else
                error('kernel_function deve essere una sottoclasse di KernelFunction');
            end

        end

        function fit(obj, X, Y)
            obj.X_train = X;
            K = obj.kernel_function.compute(X, X);

            H = [K, -K; -K, K];
            f = [obj.epsilon + Y; obj.epsilon - Y]';
            Aeq = [ones(1, length(Y)), -ones(1, length(Y))];
            beq = 0;
            
            lb = zeros(2*length(Y), 1);
            ub = obj.C * ones(2*length(Y), 1);
            
            options = optimoptions('quadprog', 'Display', 'off');
            alpha = quadprog(H, -f, [], [], Aeq, beq, lb, ub, [], options);
            
            alpha_pos = alpha(1:length(Y));
            alpha_neg = alpha(length(Y)+1:end);
            obj.alpha_svr = alpha_pos - alpha_neg;

            obj.bias = mean(Y - (K * obj.alpha_svr));
        end

        function y_pred = predict(obj, X)
            b = obj.bias;
            K_test = obj.kernel_function.compute(X, obj.X_train);
            y_pred = K_test * obj.alpha_svr + b;
        end
    end
end