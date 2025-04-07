function plot_gap(f_opt_fmincon, f_values_fmincon, f_opt_bundle, f_values_bundle)
    % f_values_fmincon: valori della funzione obiettivo per FMINCON
    % f_values_bundle: valori della funzione obiettivo per Level Bundle
    % f_opt: valore ottimo della funzione obiettivo
    
    gap_fmincon = abs(f_values_fmincon - norm(f_opt_fmincon)) / norm(f_opt_fmincon);
    gap_bundle = abs(f_values_bundle - norm(f_opt_bundle)) / norm(f_opt_bundle);
    
    figure;
    semilogy(1:length(gap_fmincon), gap_fmincon, '-o', 'LineWidth', 2, 'DisplayName', 'Oracle');
    hold on;
    semilogy(1:length(gap_bundle), gap_bundle, '-s', 'LineWidth', 2, 'DisplayName', 'Level Bundle');
    hold off;
    
    xlabel('Iterazioni');
    ylabel('Gap relativo |f_k - f^*| / |f^*|');
    title('Convergenza degli algoritmi');
    legend('Location', 'best', 'FontSize', 16);
    grid on;
end