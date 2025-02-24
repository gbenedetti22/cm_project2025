clc; clear; close all;

[X, y] = training_data("abalone");

kernel_function = SigmoidKernel();
svr = SVR(kernel_function, 1, 0.1);

fprintf("Training start..\n");
tic

svr.fit(X, y);
fprintf("Training end! :)\n");

toc

y_pred = svr.predict(X);
 
figure;
plot(X, y, 'ro', 'DisplayName', 'Dati'); hold on;
plot(X, y_pred, 'b-', 'DisplayName', 'SVR con Solver');
plot(X, K*x_k + b, 'g--', 'DisplayName', 'SVR con Level Bundle');
legend;
title('Support Vector Regression');