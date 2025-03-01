clc; clear; close all;

[X, y] = training_data("abalone");

kernel_function = GaussianKernel();
lbm = LBM(500, 1e-6, 0.1, 0.5, 20);
svr = SVR(kernel_function, 1, 0.1, lbm);

fprintf("Training start..\n");
tic

[X_sv, Y_sv] = svr.fit(X, y);
fprintf("Training end! :)\n");

toc

y_pred = svr.predict(X);

disp(mse(y_pred, y));

% figure; hold on;
% plot(X, y, '-', 'LineWidth', 1, 'DisplayName', 'Dati training');
% plot(X, y_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'Predizione SVR');
% % scatter(X_sv, Y_sv, 'ks', 'MarkerFaceColor', 'g', 'DisplayName', 'Support Vectors');
% xlabel('X'); ylabel('y'); title('SVR Duale (RBF) con Bundle Method');
% legend; grid on;
