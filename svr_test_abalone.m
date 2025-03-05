clc; clear; close all;

% Load data
[X, y] = training_data("abalone");

% Standardize features and target
[X_std, mu_X, sigma_X] = zscore(X);
[y_std, mu_y, sigma_y] = zscore(y);

% Hyperparameters
gamma_value = 1 / size(X_std, 2);  % 1/num_features
kernel_function = RBFKernel(gamma_value);
C = 10;
epsilon = 0.1;  % Now relative to standardized y

% Configure LBM optimizer
lbm = LBM(1000, 1e-5, 0.1, 0.9, 50, 20);

% Train SVR
svr = SVR_norm_kTrick(kernel_function, C, epsilon, lbm);
fprintf("Training SVR...\n");
tic
[X_sv, y_sv] = svr.fit(X_std, y_std);
training_time = toc;

% Predict and un-standardize y
y_pred_std = svr.predict(X_std);
y_pred = y_pred_std * sigma_y + mu_y;

% Metrics
mse_value = mean((y - y_pred).^2);
r2_value = 1 - sum((y - y_pred).^2) / sum((y - mean(y)).^2);

fprintf("MSE: %.4f\n", mse_value);
fprintf("R²: %.4f\n", r2_value);
fprintf("Support Vectors: %d\n", length(X_sv));