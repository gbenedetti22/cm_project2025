clc; clear; close all;

%% Dati e Parametri
[x, y] = training_data("abalone");
N = length(x);                    % numero di campioni
x = zscore (x);

% Parametri SVR
epsilon = 0.05;    % oppure prova epsilon = 0.1
C = 1;            
maxIter = 250;
tol = 1e-12;        % tol meno stringente
theta = 0.5;

% Parametri per il kernel RBF
gamma = 0.1;       % prova con gamma = 0.1 oppure gamma = 1
K = RBFKernel().compute(x, x);

% Impostazioni per MOSEK
options = mskoptimset('Display','off');

%% Inizializzazione
alpha = zeros(N,1);         % inizializzazione di alpha (∑α=0)
alpha_best = alpha;
f_best = dual_function(alpha, K, y, epsilon);

% Bundle: memorizza alpha, valore funzione e sottogradiente per ogni iterazione
bundle.alpha = [];
bundle.f = [];
bundle.g = [];

% Per il debug: memorizziamo l'evoluzione di f(alpha)
f_history = [];

%% Iterazione del Level Bundle Method
for k = 1:maxIter
    % Calcola funzione duale e sottogradiente
    [f_val, g_val] = dual_function_and_subgrad(alpha, K, y, epsilon);
    f_history = [f_history; f_val];
    
    % Stampa di debug
    fprintf('Iterazione %d, f(alpha) = %.4f, ||grad|| = %.4f\n', k, f_val, norm(g_val));
    
    % Aggiorna bundle
    bundle.alpha = [bundle.alpha, alpha];
    bundle.f = [bundle.f; f_val];
    bundle.g = [bundle.g, g_val];
    
    % Aggiorna la migliore soluzione trovata
    if f_val < f_best
        f_best = f_val;
        alpha_best = alpha;
    end
    
    % Calcola il livello: f_level = theta * f_val + (1 - theta) * f_best
    f_level = theta * f_val + (1 - theta) * f_best;
    
    % Costruisci i vincoli delle bundle cuts
    m = size(bundle.f, 1);
    A_bundle = zeros(m, N);
    b_bundle = zeros(m, 1);
    for j = 1:m
        A_bundle(j,:) = bundle.g(:,j)';
        b_bundle(j) = f_level - bundle.f(j) + bundle.g(:,j)' * bundle.alpha(:,j);
    end
    
    % Vincoli originali: ∑α = 0, -C <= alpha <= C
    Aeq = ones(1, N);
    beq = 0;
    lb = -C * ones(N,1);
    ub = C * ones(N,1);
    
    % Formulazione del master problem
    H_qp = eye(N);
    f_qp = -alpha_best;
    
    [alpha_new, ~, exitflag] = quadprog(H_qp, f_qp, A_bundle, b_bundle, Aeq, beq, lb, ub, [], options);
    
    if exitflag ~= 1
        fprintf('quadprog non ha trovato soluzione all''iterazione %d\n', k);
        alpha_new = alpha_best;
    end
    
    if norm(alpha_new - alpha) < tol
        fprintf('Convergenza raggiunta all''iterazione %d\n', k);
        alpha = alpha_new;
        break;
    end
    
    alpha = alpha_new;
end

%% Calcolo del parametro b e Predizione
sv_idx = find(abs(alpha) < C - tol);
if isempty(sv_idx)
    b = 0;
else
    b = mean( y(sv_idx) - K(sv_idx,:) * alpha );
end

y_pred = K * alpha + b;
mse = mean((y - y_pred).^2);
fprintf('MSE sui dati di training: %f\n', mse);

%% Plot dei risultati
figure;
scatter(x, y, 'b', 'filled'); hold on;
plot(x, y_pred, 'r-', 'LineWidth', 2);
xlabel('x'); ylabel('y');
title('SVR con Level Bundle Method (Kernel RBF)');
legend('Dati','Predizione');
grid on;

%% Plot dell'evoluzione di f(alpha)
figure;
plot(f_history, 'LineWidth', 2);
xlabel('Iterazione');
ylabel('f(alpha)');
title('Evoluzione della funzione obiettivo');

%% Funzioni ausiliarie

function [f_val] = dual_function(alpha, K, y, epsilon)
    % f(alpha) = 0.5*alpha'*K*alpha - y'*alpha + epsilon * sum(|alpha|)
    f_val = 0.5 * alpha' * K * alpha - y' * alpha + epsilon * sum(abs(alpha));
end

function [f_val, g_val] = dual_function_and_subgrad(alpha, K, y, epsilon)
    % Calcola f(alpha) e il sottogradiente
    f_val = 0.5 * alpha' * K * alpha - y' * alpha + epsilon * sum(abs(alpha));
    % Definisci il sottogradiente; per alpha=0, usa 0 (si può modificare per altri approcci)
    g_sign = sign(alpha);
    g_sign(alpha==0) = 0;
    g_val = K * alpha - y + epsilon * g_sign;
end
