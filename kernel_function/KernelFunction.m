classdef (Abstract) KernelFunction
    methods (Abstract)
        K = compute(obj, x, y);
    end
end