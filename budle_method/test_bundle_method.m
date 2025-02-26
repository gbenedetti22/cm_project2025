clc; clear;

f = @(x) max(x, -x - 1);
grad_f = @(x) (x > -0.5) - (x <= -0.5); % Subgradiente

x0 = -1.5;  % Punto iniziale lontano dal minimo
m1 = 0.1;
epsilon = 1e-4;
f_level = f(x0);
max_iter = 100;

[x_min, h] = level_bundle_method(f, grad_f, x0, m1, epsilon, f_level, max_iter);
disp(['Minimo trovato: ', num2str(x_min)]);


plot(h);