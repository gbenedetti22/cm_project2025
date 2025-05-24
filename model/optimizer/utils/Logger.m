% Utility class to print values in output. It's possible to disable certain
% output during object creation

classdef Logger < handle
    properties(Access=private)
        logModes        % can contain: iter | plot | summary
        plotEnabled
        iterEnabled
        summaryEnabled
        h
    end

    methods
        function obj = Logger(logString)
            obj.logModes = logString;
            obj.iterEnabled = contains(logString, "iter");
            obj.plotEnabled = contains(logString, "plot");
            obj.summaryEnabled = contains(logString, "summary");

            if obj.plotEnabled
                obj.h = animatedline('LineStyle','-', 'Marker','none', 'LineWidth', 2);
            end
        end

        function log(obj, iter, f_new, g_new, gap, f_best)
            if obj.iterEnabled
                fprintf('Iter: %d | f(x): %.6f | Grad norm: %.6e | Relative gap: %.6e\n', ...
                        iter, f_new, norm(g_new), gap);
            end

            if obj.plotEnabled
                addpoints(obj.h, iter, f_best);
                drawnow;
            end
        end

        function summary(obj, total_iters, f_final, final_gap, time)
            if obj.summaryEnabled
                fprintf('\n--- Summary ---\n');
                fprintf('Total iterations: %d\n', total_iters);
                fprintf('Execution time: %.2f seconds\n', time);
                fprintf('Best f(x): %.6f\n', f_final);
                fprintf('Final gap (UB - LB): %.6e\n', final_gap);
                fprintf('----------------\n');
            end
        end
    end
end
