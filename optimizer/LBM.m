classdef LBM    
    properties(Access=private)
        max_iter
        epsilon
        max_constraints
    end
    
    methods
        function obj = LBM(max_iter,epsilon, max_constraints)
            obj.max_iter = max_iter;
            obj.epsilon = epsilon;

            if nargin < 3
                max_constraints = inf;
            end
            
            obj.max_constraints = max_constraints;
        end
        
        function alpha = optimize(obj, Y, H, f)
            alpha = zeros(2 * length(Y), 1);
            
            B = struct('alpha', {}, 'f_alpha', {}, 'g_alpha_vec', {});
            f_level = obj.obj_function(alpha, H, f);
            m1 = 0.01;
            
            loading_bar = waitbar(0,'Computing LBM...');
            for iter = 1:obj.max_iter
                f_alpha = obj.obj_function(alpha, H, f);
                g_alpha = H * alpha - f;
                
                B(iter).alpha = alpha;
                B(iter).f_alpha = f_alpha;
                B(iter).g_alpha_vec = g_alpha;

                alpha_next = obj.project_onto_level(B, f_level, iter);
                               
                if norm(alpha_next - alpha) <= obj.epsilon
                    break;
                end

                if obj.obj_function(alpha_next, H, f) - f_alpha <= m1 * (obj.approx_f_bundle(B, alpha_next) - f_alpha)
                    alpha = alpha_next;
                    f_level = min(f_level, obj.obj_function(alpha, H, f));
                else
                    disp("no accettata");
                    f_level = 0.5 * (f_level + f_alpha);
                end
                                                
                waitbar(iter/obj.max_iter, loading_bar, "Computing LBM (" + (iter) + "/" + obj.max_iter + ")");
            end
            
            delete(loading_bar);
        end

        function f_val = obj_function(~, alpha, H, f)
            f_val = 0.5 * alpha' * H * alpha - f' * alpha;
        end

        function x_proj = project_onto_level(obj, B, f_level, iter)
            x_k = B(iter).alpha;

            A = []; b = [];

            if isinf(obj.max_constraints)
                start_idx = 1;
            else
                start_idx = max(1, iter - obj.max_constraints + 1);
            end

            for i = start_idx:iter
                x_i = B(i).alpha;
                f_i = B(i).f_alpha;
                g_i = B(i).g_alpha_vec;

                A = [A; g_i'];
                b = [b; f_level - f_i + g_i' * x_i];
            end

            n = length(x_k);
            H = eye(n);
            f = -x_k;

            options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');
            [x_proj, ~, exitflag] = quadprog(2*H, f, A, b, [], [], [], [], [], options);

            if exitflag <= 0
                warning('quadprog didnt found a solution. Keeping x_k.');
                x_proj = x_k;
            end
        end

        function f_B_x = approx_f_bundle(~, B, x)
            len = length(B);
            f_values = zeros(len, 1);
            
            for i = 1:len
                x_i = B(i).alpha;
                f_i = B(i).f_alpha;
                g_i = B(i).g_alpha_vec;
                f_values(i) = f_i + g_i' * (x - x_i);
            end
            
            f_B_x = max(f_values);
        end
    end
end