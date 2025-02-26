function [x, history] = level_bundle_method(f, grad_f, x0, m1, epsilon, f_level, max_iter)

    x = x0;
    B = { {x, f(x), grad_f(x)} };
    iter = 0;
    history = zeros(max_iter);
    
    while iter < max_iter
        iter = iter + 1;

        x_next = project_onto_level(B, f_level);

        if norm(x_next - x) <= epsilon
            break;
        end

        if f(x_next) - f(x) <= m1 * (approx_f_bundle(B, x_next) - f(x))
            x = x_next;
            f_level = min(f_level, f(x_next));
        else
            f_level = 0.5 * (f_level + f(x));
        end

        B{end+1, 1} = {x_next, f(x_next), grad_f(x_next)};
        history(iter) = x;
    end
end

function f_B_x = approx_f_bundle(B, x)
    f_values = zeros(length(B), 1);
    for i = 1:length(B)
        x_i = B{i}{1};
        f_i = B{i}{2};
        g_i = B{i}{3};
        f_values(i) = f_i + g_i' * (x - x_i);
    end
    f_B_x = max(f_values);
end

