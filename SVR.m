classdef SVR < handle
    properties(Access=private)
        kernel_function
        C double
        epsilon double
        max_iter
        alpha_svr
        bias
        X_train
        opt
        tol
        f_values
        f_times
    end

    methods
        function obj = SVR(params)
            default_params = struct(...
                'max_iter', 90, ...
                'C',        1, ...
                'epsilon',  0.05, ...
                'tol',      1e-5, ...
                'opt',      false ...
                );

            if ~isfield(params, 'kernel_function')
                error('Struct params must contain field: kernel_function');
            end

            if ~isa(params.kernel_function, 'KernelFunction')
                error('kernel_function must be a subclass of KernelFunction');
            end
            obj.kernel_function = params.kernel_function;

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



        function [x, f_values, f_times] = fit(obj, X, Y)
            obj.X_train = X;
            K = obj.kernel_function.compute(X, X);

            if isa(obj.opt, 'LBM')
                [x, f_values, f_times] = obj.opt.optimize(K, Y, obj.C, obj.max_iter, obj.epsilon);
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
                    'SpecifyObjectiveGradient', true, 'MaxIterations', obj.max_iter, ...
                    'OutputFcn', outHandle);

                x = fmincon(svr_dual, alpha0, A, b, Aeq, beq, lb, ub, [], options);
                f_values = obj.f_values;
                f_times = obj.f_times;
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
                obj.f_times = [];
                tic
            end

            if strcmp(state, 'iter')
                obj.f_values = [obj.f_values; optimValues.fval];
                obj.f_times = [obj.f_times; toc];
            end

            stop = false;
        end

    end
end