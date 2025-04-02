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
        f_values
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

        function [x, f_values] = fit(obj, X, Y)
            obj.X_train = X;
            K = obj.kernel_function.compute(X, X);

            if isa(obj.opt, 'LBM')
                [x, f_values] = obj.opt.optimize(K, Y, obj.C);
            else
                n = length(Y);

                svr_dual =  @(x) obj.svr_dual_function(x, K, Y, obj.epsilon);
               
                alpha0 = zeros(n, 1);

                Aeq = ones(1, n);
                beq = 0;

                A = [];
                b = [];

                lb = -obj.C * ones(n, 1);
                ub = obj.C * ones(n, 1);

                outHandle = @(x, optimValues, state) obj.outfun(x, optimValues, state);

                % 'PlotFcn', {@optimplotfval}, ...
                options = optimoptions('fmincon', 'Display', 'iter', ...
                    'SpecifyObjectiveGradient', true, 'MaxIterations', 50, ...
                    'OutputFcn', outHandle);

                x = fmincon(svr_dual, alpha0, A, b, Aeq, beq, lb, ub, [], options);
                f_values = obj.f_values;
            end
            
            obj.alpha_svr = x;
            
            sv_indices = find(abs(obj.alpha_svr) > obj.tol);
            
            if isempty(sv_indices)
                warning("Support vectors not found");

                obj.bias = mean(Y - (K * obj.alpha_svr));
            else
                obj.bias = mean(Y(sv_indices) - K(sv_indices, :) * obj.alpha_svr);
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

        function stop = outfun(obj, ~, optimValues, state)
            if strcmp(state, 'init')
                obj.f_values = [];
            end

            if strcmp(state, 'iter')
                obj.f_values = [obj.f_values; optimValues.fval];
            end

            stop = false;
        end

    end
end