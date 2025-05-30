% Utility function for comparing Oracle and LBM in a time-based context.
% Specifically, the plot shows what happens at the same time *t*, making it
% useful for observing the function values at a specific moment

% If times are different, interpolation is used for aligning

function plot_time(f_values_oracle, f_time_oracle, f_values_bundle, f_time_bundle)
n = max(length(f_time_oracle), length(f_time_bundle));

common_time = linspace(0, min(f_time_oracle(end), f_time_bundle(end)), n);

oracle_interp = interp1(f_time_oracle, f_values_oracle, common_time, 'linear', 'extrap');
bundle_interp = interp1(f_time_bundle, f_values_bundle, common_time, 'linear', 'extrap');

figure;
plot(common_time, oracle_interp, '-o', 'DisplayName', 'Oracle', 'LineWidth', 1.5);
hold on;
plot(common_time, bundle_interp, '-x', 'DisplayName', 'SVR with LBM', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Value');
title('Objective function values based on time', 'FontSize', 18);
legend;
grid on;
end