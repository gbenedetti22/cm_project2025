% Oracle implementation that use the dual formulation of a classical SVR

classdef Oracle
    methods
        function [x, history] = optimize(~, K, Y, C, max_iter, epsilon)
            % Convex QP and non-differentiable optimization problem using MOSEK
            %
            % Inputs:
            %   K         - kernel matrix
            %   Y         - target values
            %   C         - regularization parameter
            %   max_iter  - maximum number of iterations
            %   epsilon   - tolerance
            %
            % Outputs:
            %   x         - optimal solution
            %   history   - optimization history object

            %% 1. Initialization
            n = length(Y);
            Q = sparse(K);

            %% 2. Constraints
            Aeq = ones(1, n);            % Equality constraint: sum(x) = 0
            beq = 0;
            lb = -C * ones(n, 1);        % Lower bounds on x
            ub =  C * ones(n, 1);        % Upper bounds on x

            % Linear equality constraint Ax = b
            prob.a = sparse(Aeq);
            prob.blc = beq;
            prob.buc = beq;

            % Box constraints on x
            prob.blx = lb;
            prob.bux = ub;

            %% 3. MOSEK problem setup
            % Only the lower triangle of Q is passed, in accordance with
            % MOSEK doc
            [idx_i, idx_j, vals] = find(tril(Q));
            prob.qosubi = idx_i;
            prob.qosubj = idx_j;
            prob.qoval  = vals;

            c = -Y + epsilon;   % Linear term
            prob.c = c;

            %% 4. Optimization history
            history = OptHistory(max_iter);

            %% 5. Solve the problem using MOSEK
            [~, res] = mosekopt('symbcon');
            symbcon = res.symbcon;

            % Callback defined in the oracle_callback.m file
            callback.iter = 'oracle_callback';
            callback.iterhandle = struct(...
                'symbcon', symbcon, ...
                'history', history ...
                );

            param.MSK_IPAR_INTPNT_MAX_ITERATIONS = max_iter;

            [~, res] = mosekopt('minimize info', prob, param, callback);

            %% 6. Checking results
            x = res.sol.itr.xx;

            disp("Minimum Found: " + res.sol.itr.pobjval);
            disp("Response Code: " + res.rcode);

            % relative gap between primal and dual solutions
            pd_gap = abs(res.sol.itr.pobjval - res.sol.itr.dobjval) / abs(res.sol.itr.dobjval);
            disp("Primal-dual gap: " + pd_gap);
        end


    end
end