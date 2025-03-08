classdef GaussianKernel < KernelFunction
    properties (Access = private)
        sigma double
    end
    
    methods
        function obj = GaussianKernel(sigma)
            if nargin < 1, sigma = 1; end
            obj.sigma = sigma;
        end
        
        function K = compute(obj, X, Y)
            % Calcola la distanza euclidea al quadrato tra tutte le righe di X e Y
            %||x - y||² = (x - y)ᵀ(x - y) = ||x||² + ||y||² - 2x·y
            X_sq = sum(X.^2, 2);
            Y_sq = sum(Y.^2, 2);
            XY = X * Y';
            % Calcola la distanza Euclidea al quadrato per tutte le coppie (i,j)
            % Ogni elemento (i,j) è: X_sq(i) - 2*XY(i,j) + Y_sq(j)
            squared_dist = X_sq - 2*XY + Y_sq';
            
            % Applico formula: K(x,y) = exp(-||x - y||² / (2σ²))
            K = exp(-squared_dist / (2 * obj.sigma^2));
        end
    end
end