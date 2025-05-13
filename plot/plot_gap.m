function plot_gap(f_values_oracle, f_values_bundle)

    % Trova numero di iterazioni
    n_oracle = length(f_values_oracle);
    n_bundle = length(f_values_bundle);
    n_common = min(n_oracle, n_bundle);

    % Campionamento uniforme
    idx_oracle = round(linspace(1, n_oracle, n_common));
    idx_bundle = round(linspace(1, n_bundle, n_common));

    f_sampled_oracle = f_values_oracle(idx_oracle);
    f_sampled_bundle = f_values_bundle(idx_bundle);

    % Minimi locali
    f_star_oracle = min(f_sampled_oracle);
    f_star_bundle = min(f_sampled_bundle);

    % Gap relativi
    gap_oracle = (abs(f_sampled_oracle) - abs(f_star_oracle)) / abs(f_star_oracle);
    gap_bundle = (abs(f_sampled_bundle) - abs(f_star_bundle)) / abs(f_star_bundle);

    % Plot
    figure;
    semilogy(1:n_common, gap_oracle, '-o', 'LineWidth', 2, 'DisplayName', 'Oracle');
    hold on;
    semilogy(1:n_common, gap_bundle, '-s', 'LineWidth', 2, 'DisplayName', 'Level Bundle');
    hold off;

    xlabel('Iterations (normalized)');
    ylabel('Relative Gap');
    legend('Location', 'best', 'FontSize', 16);
    grid on;
end

% function plot_gap(f_values_oracle, f_values_bundle)
% 
%     f_star_oracle = min(f_values_oracle);
%     f_star_bundle = min(f_values_bundle);
% 
%     gap_oracle = abs(f_values_oracle) - abs(f_star_oracle) / abs(f_star_oracle);
%     gap_bundle = abs(f_values_bundle) - abs(f_star_bundle) / abs(f_star_bundle);
% 
%     figure;
%     semilogy(1:length(gap_oracle), gap_oracle, '-o', 'LineWidth', 2, 'DisplayName', 'Oracle');
%     hold on;
%     semilogy(1:length(gap_bundle), gap_bundle, '-s', 'LineWidth', 2, 'DisplayName', 'Level Bundle');
%     hold off;
% 
%     xlabel('Iterations');
%     ylabel('Relative Gap');
%     legend('Location', 'best', 'FontSize', 16);
%     grid on;
% end