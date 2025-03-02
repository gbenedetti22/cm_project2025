% Clear workspace and figures
clc; clear; close all;

% Load abalone dataset
[X, y] = training_data("abalone");

% Set up SVR with Bundle Method
gamma_value = 0.1;
kernel_function = RBFKernel(gamma_value);

lbm = LBM(500, 1e-6, 0.1, 0.5, 20, 10);
svr = SVR(kernel_function, 1, 0.1, lbm);

fprintf("Training SVR model...\n");
tic
[X_sv, y_sv] = svr.fit(X, y);
training_time = toc;
fprintf("Training completed in %.2f seconds\n", training_time);

% Get predictions
y_pred = svr.predict(X);
mse_value = mse(y_pred, y);
fprintf("MSE: %.4f\n", mse_value);

%% VISUALIZATION 1: SVR PREDICTION AND SUPPORT VECTORS
figure('Position', [100, 100, 900, 500], 'Name', 'SVR Predictions & Support Vectors');

scatter(X, y, 10, 'b.', 'MarkerEdgeAlpha', 0.3); hold on;
plot(X, y_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'SVR Prediction');
scatter(X_sv, y_sv, 40, 'gs', 'MarkerEdgeColor', 'k', 'LineWidth', 1, 'DisplayName', 'Support Vectors');

xlabel('Feature Value');
ylabel('Target Value');
title(sprintf('SVR Prediction (MSE: %.4f) with %d Support Vectors', mse_value, length(X_sv)));
legend('Data Points', 'SVR Prediction', 'Support Vectors');
grid on;

%% VISUALIZATION 2: ERROR DISTRIBUTION & ACCURACY
figure('Position', [100, 100, 900, 400], 'Name', 'SVR Error Analysis');

errors = y - y_pred;

subplot(1,2,1);
histogram(errors, 30, 'FaceColor', [0.3, 0.6, 0.9], 'EdgeColor', 'w');
xlabel('Prediction Error');
ylabel('Frequency');
title('Error Distribution');
grid on;

subplot(1,2,2);
scatter(y, y_pred, 15, 'b.', 'MarkerEdgeAlpha', 0.3);
hold on;
min_val = min([y; y_pred]);
max_val = max([y; y_pred]);
plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
xlabel('Actual Values');
ylabel('Predicted Values');
title('Actual vs. Predicted');
axis equal; grid on;

%% VISUALIZATION 3: EPSILON-TUBE VISUALIZATION
figure('Position', [100, 100, 900, 400], 'Name', 'SVR Epsilon-Tube Visualization');

epsilon = 0.1; % From SVR model

scatter(X, y, 10, 'b.', 'MarkerEdgeAlpha', 0.2);
hold on;
plot(X, y_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'SVR Prediction');
plot(X, y_pred + epsilon, 'r--', 'LineWidth', 1, 'DisplayName', 'Epsilon-Tube');
plot(X, y_pred - epsilon, 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');

scatter(X_sv, y_sv, 40, 'gs', 'MarkerEdgeColor', 'k', 'LineWidth', 1, 'DisplayName', 'Support Vectors');

xlabel('Feature Value');
ylabel('Target Value');
title(sprintf('SVR Prediction with \\epsilon-Tube (\\epsilon = %.2f)', epsilon));
legend('Data Points', 'SVR Prediction', 'Epsilon Tube', 'Support Vectors', 'Location', 'best');
grid on;

%% VISUALIZATION 4: LBM CONVERGENCE ANALYSIS
if isfield(lbm, 'f_history') && ~isempty(lbm.f_history)
    figure('Position', [100, 100, 900, 400], 'Name', 'LBM Convergence Analysis');

    subplot(1,2,1);
    plot(lbm.f_history, 'b-', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Objective Value');
    title('Objective Function Convergence');
    grid on;

    if isfield(lbm, 'step_history') && ~isempty(lbm.step_history)
        subplot(1,2,2);
        semilogy(lbm.step_history, 'r-', 'LineWidth', 2);
        xlabel('Iteration');
        ylabel('Step Size (log scale)');
        title('Step Size Convergence');
        grid on;
    end
end

%% MODEL SUMMARY BOX
figure('Position', [100, 100, 400, 200], 'Name', 'Model Summary');

text_str = sprintf(['Model Information:\n', ...
                   '- Kernel: RBF\n', ...
                   '- C = %.2f\n', ...
                   '- Epsilon = %.2f\n', ...
                   '- Support Vectors: %d (%.1f%%)\n', ...
                   '- MSE: %.4f\n', ...
                   '- Training Time: %.2f sec'], ...
                   1, epsilon, length(X_sv), 100*length(X_sv)/length(X), mse_value, training_time);
                   
annotation('textbox', [0.1, 0.1, 0.8, 0.8], 'String', text_str, ...
           'EdgeColor', 'none', 'BackgroundColor', [0.95, 0.95, 0.95], ...
           'FontSize', 10, 'FontWeight', 'bold');

