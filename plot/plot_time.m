function plot_time(f_values_fmincon, f_time_fmincon, f_values_bundle, f_time_bundle)
n = max(length(f_time_fmincon), length(f_time_bundle));

common_time = linspace(0, min(f_time_fmincon(end), f_time_bundle(end)), n);

fmincon_interp = interp1(f_time_fmincon, f_values_fmincon, common_time, 'linear', 'extrap');
bundle_interp = interp1(f_time_bundle, f_values_bundle, common_time, 'linear', 'extrap');

figure;
plot(common_time, fmincon_interp, '-o', 'DisplayName', 'Oracle', 'LineWidth', 1.5);
hold on;
plot(common_time, bundle_interp, '-x', 'DisplayName', 'SVR with LBM', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Value');
legend('Location', 'best', 'FontSize', 16);
grid on;
end
