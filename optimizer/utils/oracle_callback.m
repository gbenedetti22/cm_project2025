function [stop] = oracle_callback(handle, where, info)
    stop = 0;

    symbcon = handle.symbcon;
    
    if where == symbcon.MSK_CALLBACK_INTPNT
        iter = info.MSK_IINF_INTPNT_ITER;
        
        handle.history.f_values(iter + 1) = info.MSK_DINF_INTPNT_PRIMAL_OBJ;
        handle.history.f_times(iter + 1) = info.MSK_DINF_OPTIMIZER_TIME;
    end

end