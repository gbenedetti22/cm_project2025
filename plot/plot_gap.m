% Utility function that displays the (relative) gap during iterations. 
% If the values differ, those with higher values are reduced by sampling 
% based on the ones with lower values

function plot_gap(f_values_oracle, f_values_bundle)
    n_oracle = length(f_values_oracle);
    n_bundle = length(f_values_bundle);
    n_common = min(n_oracle, n_bundle);

    idx_oracle = round(linspace(1, n_oracle, n_common));
    idx_bundle = round(linspace(1, n_bundle, n_common));

    f_sampled_oracle = f_values_oracle(idx_oracle);
    f_sampled_bundle = f_values_bundle(idx_bundle);

    f_star = min(f_sampled_oracle);

    rel_gap = (f_sampled_bundle - f_sampled_oracle) / f_star;

    figure;
    semilogy(1:n_common, abs(rel_gap), '-o', 'LineWidth', 2, 'DisplayName', 'Relative Gap');
    xlabel('Iterations');
    ylabel('Relative Gap');
    title("Relative Gap (Optimal Solution)", 'FontSize', 18);
    legend('Location', 'best', 'FontSize', 16);
    grid on;
end