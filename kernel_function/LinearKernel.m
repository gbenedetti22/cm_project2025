classdef LinearKernel < KernelFunction
    methods
        % Implementazione del metodo compute
        function K = compute(~, x, y)
            K = x * y';
        end
    end
end