clc; clear; close all;

[X, y] = training_data("abalone");
X = zscore(X);

% Parametri SVR
epsilon = 0.05;
C = 1;
maxIter = 200;
tol = 1e-11;
theta = 0.9;
max_constraint = 50;

% Parametri per il kernel RBF
sigma = 0.7;

kernel_function = RBFKernel(sigma);

lbm = LBM(maxIter, epsilon, tol, theta, max_constraint);
svr = SVR(kernel_function, C, epsilon, lbm);

fprintf("Training start..\n");
tic

[X_sv, Y_sv] = svr.fit(X, y);
fprintf("Training end! :) \n");

toc

y_pred = svr.predict(X);

disp(mse(y_pred, y));

% figure; hold on;
% plot(X, y, '-', 'LineWidth', 1, 'DisplayName', 'Training data');
% plot(X, y_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'SVR Predictions');
% scatter(X_sv, Y_sv, 'ks', 'MarkerFaceColor', 'g', 'DisplayName', 'Support Vectors');
% xlabel('X'); ylabel('y'); title('SVR (with RBF Kernel) using LBM', 'FontSize', 22);
% legend('FontSize', 18); grid on;