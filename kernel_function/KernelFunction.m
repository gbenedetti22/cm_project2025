% General class that is implemented by every Kernel Function.
% This abstract class represent a general Kernel Function
classdef (Abstract) KernelFunction
    methods (Abstract)
        K = compute(obj, x, y);
    end
end