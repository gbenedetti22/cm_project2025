classdef Oracle
    properties(Access=private)

    end

    methods

        function [x, history] = optimize(~, K, Y, C, max_iter, epsilon)
            n = length(Y);
            Q = sparse(K);
            c = -Y + epsilon;
            Aeq = ones(1, n);
            beq = 0;
            lb = -C * ones(n, 1);
            ub = C * ones(n, 1);

            prob.c = c;

            [idx_i, idx_j, vals] = find(tril(Q));
            prob.qosubi = idx_i;
            prob.qosubj = idx_j;
            prob.qoval  = vals;

            prob.a = sparse(Aeq);
            prob.blc = beq;
            prob.buc = beq;
            prob.blx = lb;
            prob.bux = ub;

            history = OptHistory(max_iter);

            [~, res] = mosekopt('symbcon');
            symbcon = res.symbcon;

            callback.iter = 'oracle_callback';
            callback.iterhandle = struct(...
                'symbcon', symbcon, ...
                'history', history ...
                );

            param.MSK_IPAR_INTPNT_MAX_ITERATIONS = max_iter;

            [~, res] = mosekopt('minimize info', prob, param, callback);

            x = res.sol.itr.xx;

            disp("Minimum Found: " + res.sol.itr.pobjval);
            disp("Response Code: " + res.rcode);
            pd_gap = abs(res.sol.itr.pobjval - res.sol.itr.dobjval);
            disp("Gap primale-duale: " + pd_gap);
        end

    end
end