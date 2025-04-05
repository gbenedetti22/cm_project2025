clc; clear; close all;

[X, y] = training_data("abalone");
X = zscore(X);

% Parametri SVR
epsilon = 0.05;
C = 1;
maxIter = 50;
tol = 1e-2;
theta = 0.9;
max_constraint = 50;

% Parametri per il kernel RBF
sigma = 0.5;

kernel_function = RBFKernel(sigma);

lbm = LBM(maxIter, epsilon, tol, theta, max_constraint);

svr = SVR(kernel_function, C, epsilon);
svr_lbm = SVR(kernel_function, C, epsilon, lbm);

[x, f_values] = svr.fit(X, y);
[x_lbm, f_values_lbm] = svr_lbm.fit(X, y);

y_pred = svr.predict(X);
y_pred_lbm = svr_lbm.predict(X);
disp("MSE: " + mse(y_pred, y));
disp("MSE (LBM): " + mse(y_pred_lbm, y));

plot_gap(x, f_values, x_lbm, f_values_lbm);