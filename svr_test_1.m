clc; clear; close all;

[X, y] = training_data("sin");

kernel_function = RBFKernel();
svr = SVR(kernel_function, 1, 0.1, LBM(500, 1e-6));

fprintf("Training start..\n");
tic

svr.fit(X, y);
fprintf("Training end! :)\n");

toc

y_pred = svr.predict(X);

figure;
plot(X, y, 'ro', 'DisplayName', 'Dati'); hold on;
plot(X, y_pred, 'b-', 'DisplayName', 'SVR con Solver');
legend;
title('Support Vector Regression');