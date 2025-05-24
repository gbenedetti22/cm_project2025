classdef GaussianKernel < KernelFunction
    properties (Access = private)
        sigma double
    end

    methods
        function obj = GaussianKernel(sigma)
            if nargin < 1
                sigma = 1;
            end
            obj.sigma = sigma;
        end

        function K = compute(obj, x, y)
            K = exp(-norm(x - y)^2 / (2 * obj.sigma^2));
        end
    end
end