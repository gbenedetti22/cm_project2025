function plot_time(f_values_fmincon, f_time_fmincon, f_values_bundle, f_time_bundle)
    n = max(length(f_time_fmincon), length(f_time_bundle));
    common_time = linspace(0, min(f_time_fmincon(end), f_time_bundle(end)), n);
    
    % Clean bundle data: Remove any non-finite values
    validIdx = isfinite(f_time_bundle) & isfinite(f_values_bundle);
    f_time_bundle_clean = f_time_bundle(validIdx);
    f_values_bundle_clean = f_values_bundle(validIdx);
    
    fmincon_interp = interp1(f_time_fmincon, f_values_fmincon, common_time, 'linear', 'extrap');
    bundle_interp = interp1(f_time_bundle_clean, f_values_bundle_clean, common_time, 'linear', 'extrap');
    
    figure;
    plot(common_time, fmincon_interp, '-o', 'DisplayName', 'Oracle', 'LineWidth', 1.5);
    hold on;
    plot(common_time, bundle_interp, '-x', 'DisplayName', 'SVR with LBM', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Value');
    legend;
    grid on;
end
