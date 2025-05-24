classdef LinearKernel < KernelFunction
    methods
        function K = compute(~, x, y)
            K = x * y';
        end
    end
end