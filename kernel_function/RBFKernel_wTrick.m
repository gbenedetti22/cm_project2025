%%% RBF with KERNEL TRICK 
classdef RBFKernel_wTrick < KernelFunction
    properties
        gamma  % parametro per il kernel RBF, ad esempio gamma = 1/(2*sigma^2)
    end

    methods
        function obj = RBFKernel_wTrick(gamma)
            if nargin < 1
                obj.gamma = 0.1; % valore di default, da regolare in base ai dati
            else
                obj.gamma = gamma;
            end
        end

        function K = compute(obj, X1, X2)
            % Calcola la matrice kernel RBF tra X1 e X2
            % X1: n1 x d, X2: n2 x d
            % K: n1 x n2
            X1_sq = sum(X1.^2, 2);
            X2_sq = sum(X2.^2, 2);
            % bsxfun applica la somma in maniera vettoriale
            K = exp(-obj.gamma * (bsxfun(@plus, X1_sq, X2_sq') - 2 * (X1 * X2')));
        end
    end
end