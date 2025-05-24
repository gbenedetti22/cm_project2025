function analyze_results(X, y, svr, svr_lbm, h, h_lbm)

    % Extract objective values and timing history from Oracle and LBM
    f_values = h.f_values;
    f_times = h.f_times;
    f_values_lbm = h_lbm.f_values;
    f_times_lbm = h_lbm.f_times;

    % Predict targets using both models
    y_pred = svr.predict(X);
    y_pred_lbm = svr_lbm.predict(X);

    % Compute mean squared error (MSE) for both models
    mse_std = mse(y_pred, y);
    mse_lbm = mse(y_pred_lbm, y);

    % Compute relative gap between the best values of both models
    gap = (abs(min(f_values_lbm)) - abs(min(f_values))) / abs(min(f_values));
    
    
    disp("MSE: " + mse(y_pred, y));
    disp("MSE (LBM): " + mse(y_pred_lbm, y));
    disp("Relative Gap: " + gap);
    fprintf("Time (STD): %.2f sec\n", f_times(end));
    fprintf("Time (LBM): %.2f sec\n", f_times_lbm(end));
    
    % Plot convergence behavior and time performance of both models
    plot_gap(f_values, f_values_lbm);
    plot_time(f_values, f_times, f_values_lbm, f_times_lbm);
end
