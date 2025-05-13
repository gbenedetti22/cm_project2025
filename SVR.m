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

        function [x, history] = fit(obj, X, Y)
            obj.X_train = X;
            K = obj.kernel_function.compute(X, X);

            if isa(obj.opt, 'LBM')
                [x, history] = obj.opt.optimize(K, Y, obj.C, obj.max_iter, obj.epsilon);
            else
                [x, history] = Oracle().optimize(K, Y, obj.C, obj.max_iter, obj.epsilon);
            end

            obj.alpha_svr = x;

            sv_indices = find(abs(obj.alpha_svr) > obj.tol);

            if isempty(sv_indices)
                warning("Support vectors not found");

                obj.bias = mean(Y - (K * obj.alpha_svr));
            else
                obj.bias = mean(Y(sv_indices) - K(sv_indices, :) * obj.alpha_svr);
            end


            history.f_values = history.f_values';
            history.f_values = history.f_values(~isnan(history.f_values));

            history.f_times = history.f_times';
            history.f_times = history.f_times(~isnan(history.f_times));
        end

        function y_pred = predict(obj, X)
            K_test = obj.kernel_function.compute(X, obj.X_train);
            y_pred = K_test * obj.alpha_svr + obj.bias;
        end

    end
end