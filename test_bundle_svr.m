clear; clc; close all;

kernel = RBFKernel(0.1);
C = 1;
epsilon = 0.1;

svr = BundleSVR(kernel, C, epsilon);

n = 50;
X = linspace(-1, 1, n)';
Y = sin(pi*X) + 0.1*randn(n, 1);

%training utilizzando il Bundle method
svr.fit_bundle(X, Y);

Y_pred = svr.predict(X);

% Plot dei dati reali e della previsione
figure;
plot(X, Y, 'bo', 'MarkerFaceColor', 'b', 'DisplayName', 'Dati Reali'); hold on;
plot(X, Y_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'Previsione SVR (Bundle)');
xlabel('Input X');
ylabel('Output Y');
title('SVR con Bundle Method');
legend;
grid on;
