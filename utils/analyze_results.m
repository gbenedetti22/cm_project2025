function analyze_results(X, y, svr, svr_lbm, h, h_lbm)

    % Extract objective values and timing history from Oracle and LBM
    f_values = h.f_values;
    f_times = h.f_times;
    f_values_lbm = h_lbm.f_values;
    f_times_lbm = h_lbm.f_times;

    % Predict targets using both models
    y_pred = svr.predict(X);
    y_pred_lbm = svr_lbm.predict(X);

    % Compute relative gap between the best values of both models
    gap = (abs(min(f_values_lbm)) - abs(min(f_values))) / abs(min(f_values));
    
    % Print performance comparison table
    fprintf('\n\n');
    results = table;

    results.MSE = [mse(y_pred, y), mse(y_pred_lbm, y)];
    results.Min_Value = [min(f_values), min(f_values_lbm)];
    results.Relative_Gap = abs(gap);
    results.Runtime_sec = [f_times(end), f_times_lbm(end)];

    results.Properties.VariableNames = {...
        'MSE [Oracle vs LBM]', ...
        'Minimum Value [Oracle vs LBM]', ...
        'Relative Gap', ...
        'Time (sec) [Oracle vs LBM]', ...
        };

    disp(results);

    % Plot convergence behavior and time performance of both models
    plot_gap(f_values, f_values_lbm);
    plot_time(f_values, f_times, f_values_lbm, f_times_lbm);
end
