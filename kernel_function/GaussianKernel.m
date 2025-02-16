classdef GaussianKernel < KernelFunction
    properties (Access = private)
        sigma double  % Parametro sigma del kernel Gaussiano
    end

    methods
        % Costruttore
        function obj = GaussianKernel(sigma)
            if nargin < 1
                sigma = 1;  % Valore di default
            end
            obj.sigma = sigma;
        end

        % Implementazione del metodo compute
        function K = compute(obj, x, y)
            K = exp(-norm(x - y)^2 / (2 * obj.sigma^2));
        end
    end
end