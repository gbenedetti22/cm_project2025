clc; clear; close all;

[X, y] = training_data("sin");

kernel_function = RBFKernel();
lbm = LBM(500, 1e-6, 0.1, 0.5, 20, 80);
svr = SVR(kernel_function, 1, 0.1, lbm);
% lbm = LBM(1000, 1e-6, 1e-7, 0.7, 150, 50);
% svr = SVR(kernel_function, 250, 0.01, lbm);
fprintf("Training start..\n");
tic

[X_sv, Y_sv] = svr.fit(X, y);
fprintf("Training end! :)\n");

toc

y_pred = svr.predict(X);

disp(mse(y_pred, y));

figure; hold on;
plot(X, y, '-', 'LineWidth', 1, 'DisplayName', 'Dati training');
plot(X, y_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'Predizione SVR');
scatter(X_sv, Y_sv, 'ks', 'MarkerFaceColor', 'g', 'DisplayName', 'Support Vectors');
xlabel('X'); ylabel('y'); title('SVR Duale (RBF) con Bundle Method');
legend; grid on;
