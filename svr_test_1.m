clc; clear; close all;

t = linspace(-2, 2, 100)';
y = sin(3*t) + 0.1 * randn(size(t));
X = t;

kernel_function = RBFKernel(0.5);
svr = SVR(kernel_function, 1, 0.1);
svr.fit(X, y);

y_pred = svr.predict(X);

figure;
plot(X, y, 'ro', 'DisplayName', 'Dati'); hold on;
plot(X, y_pred, 'b-', 'DisplayName', 'SVR con Solver');
% plot(X, K*x_k + b, 'g--', 'DisplayName', 'SVR con Level Bundle');
legend;
title('Support Vector Regression');