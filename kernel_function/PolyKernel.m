classdef PolyKernel < KernelFunction
    properties (Access = private)
        c double
        d double
    end

    methods
        function obj = PolyKernel(c, d)
            if nargin < 1
                c = 1;
            end
            if nargin < 2
                d = 2;
            end
            obj.c = c;
            obj.d = d;
        end

        function K = compute(obj, x, y)
            K = (x * y' + obj.c) ^ obj.d;
        end
    end
end