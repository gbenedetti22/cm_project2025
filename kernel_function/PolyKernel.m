classdef PolyKernel < KernelFunction
    properties (Access = private)
        c double  % Parametro c del kernel polinomiale
        d double  % Parametro d (grado) del kernel polinomiale
    end

    methods
        % Costruttore
        function obj = PolyKernel(c, d)
            if nargin < 1
                c = 1;  % Valore di default
            end
            if nargin < 2
                d = 2;  % Valore di default
            end
            obj.c = c;
            obj.d = d;
        end

        % Implementazione del metodo compute
        function K = compute(obj, x, y)
            K = (x * y' + obj.c) ^ obj.d;
        end
    end
end