classdef RBFKernel < KernelFunction
    properties (Access = private)
        sigma double  % Parametro sigma del kernel RBF
    end

    methods
        % Costruttore
        function obj = RBFKernel(sigma)
            if nargin < 1
                sigma = 0.5;
            end
            obj.sigma = sigma;
        end

        % Implementazione del metodo compute
        function K = compute(obj, x, y)
            K = exp(-pdist2(x, y, 'euclidean').^2 / (2 * obj.sigma^2));

            if ~obj.ispsd(K)
                epsilon = 1e-8;
                K = K + epsilon * eye(size(K));
            end
        end

        function ispsd = ispsd(~, K)
            try chol(K);
                ispsd = true;
            catch ME
                ispsd = false;
            end
        end
    end
end