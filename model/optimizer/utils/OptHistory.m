% Object used to store in one place, all the values ​​useful for debugging

classdef OptHistory < handle
    properties
        f_values = [];
        f_times = [];
    end
    
    methods
        function obj = OptHistory(max_iter)
            if nargin > 0
                obj.f_values = nan(1, max_iter);
                obj.f_times = nan(1, max_iter);
            end
        end
    end
end