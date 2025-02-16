classdef SigmoidKernel < KernelFunction
    properties (Access = private)
        alpha double  % Parametro alpha del kernel sigmoidale
        c double      % Parametro c del kernel sigmoidale
    end

    methods
        % Costruttore
        function obj = SigmoidKernel(alpha, c)
            if nargin < 1
                alpha = 1;  % Valore di default
            end
            if nargin < 2
                c = 0;  % Valore di default
            end
            obj.alpha = alpha;
            obj.c = c;
        end

        % Implementazione del metodo compute
        function K = compute(obj, x, y)
            K = tanh(obj.alpha * (x * y') + obj.c);
        end
    end
end