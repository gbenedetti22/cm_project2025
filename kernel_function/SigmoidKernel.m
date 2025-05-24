classdef SigmoidKernel < KernelFunction
    properties (Access = private)
        alpha double
        c double
    end

    methods
        function obj = SigmoidKernel(alpha, c)
            if nargin < 1
                alpha = 1;
            end
            if nargin < 2
                c = 0;
            end
            obj.alpha = alpha;
            obj.c = c;
        end

        function K = compute(obj, x, y)
            K = tanh(obj.alpha * (x * y') + obj.c);
        end
    end
end