% Genera dati di esempio
X = linspace(-2, 2, 100)';
Y = sin(3*X) + 0.1 * randn(size(X));

% Normalizza i dati - questo può migliorare significativamente la convergenza
X_mean = mean(X);
X_std = std(X);
X_norm = (X - X_mean) / X_std;

Y_mean = mean(Y);
Y_std = std(Y);
Y_norm = (Y - Y_mean) / Y_std;

% Parametri SVR
C = 10;        % Parametro di regolarizzazione
epsilon = 0.1; % Parametro di insensibilità
gamma = 2;     % Parametro del kernel RBF

% Calcola la matrice del kernel RBF
n = length(X);
K = zeros(n, n);
for i = 1:n
    for j = 1:n
        K(i,j) = exp(-gamma * norm(X_norm(i) - X_norm(j))^2);
    end
end

% Controlla che K sia definita positiva (importante per la convergenza)
[~, p] = chol(K);
if p > 0
    disp('Attenzione: la matrice K non è definita positiva!');
    % Aggiungi un piccolo offset alla diagonale
    K = K + 1e-6 * eye(n);
end

% Prepara la matrice H e il vettore f per LBM
% Per SVR con kernel RBF
H = [K, -K; -K, K];
f = [epsilon * ones(n, 1) + Y_norm; epsilon * ones(n, 1) - Y_norm];

% Verifica che H sia simmetrica e semi-definita positiva
if ~issymmetric(H)
    disp('Attenzione: H non è simmetrica!');
    H = (H + H')/2;  % Forza la simmetria
end

% Verifica che H sia definita positiva (cruciale per la convergenza)
[~, p] = chol(H);
if p > 0
    disp('Attenzione: H non è definita positiva!');
    % Aggiungi un piccolo offset alla diagonale
    H = H + 1e-6 * eye(size(H, 1));
end

% Crea e configura LBM con parametri più conservativi
max_iter = 500;         % Aumenta le iterazioni
epsilon_lbm = 1e-5;     % Riduce epsilon per maggiore precisione
max_constraints = 30;   % Ottimizza il numero di vincoli

lbm = LBM(max_iter, epsilon_lbm, max_constraints);

% Esegui l'ottimizzazione
alpha = lbm.optimize(Y_norm, H, f);

% Estrai i moltiplicatori di Lagrange
alpha_plus = alpha(1:n);
alpha_minus = alpha(n+1:end);
lambda = alpha_plus - alpha_minus;

% Identifica i support vector (punti con lambda non zero)
sv_indices = find(abs(lambda) > 1e-4);
fprintf('Numero di support vector: %d su %d punti\n', length(sv_indices), n);

% Calcola b (bias)
b = 0;
for i = sv_indices'
    b = b + Y_norm(i);
    for j = 1:n
        b = b - lambda(j) * K(i,j);
    end
end
if ~isempty(sv_indices)
    b = b / length(sv_indices);
end

% Predici i valori su una griglia più fitta per la visualizzazione
X_test = linspace(-2, 2, 200)';
X_test_norm = (X_test - X_mean) / X_std;
Y_pred_norm = zeros(size(X_test));

for i = 1:length(X_test)
    for j = 1:n
        % Calcola il kernel tra il punto di test e tutti i punti di training
        k_ij = exp(-gamma * norm(X_test_norm(i) - X_norm(j))^2);
        Y_pred_norm(i) = Y_pred_norm(i) + lambda(j) * k_ij;
    end
    Y_pred_norm(i) = Y_pred_norm(i) + b;
end

% Denormalizza la predizione
Y_pred = Y_pred_norm * Y_std + Y_mean;

% Visualizza i risultati
figure;
hold on;
scatter(X, Y, 'ro', 'DisplayName', 'Dati');
plot(X_test, Y_pred, 'b-', 'LineWidth', 2, 'DisplayName', 'SVR con LBM');
plot(X_test, sin(3*X_test), 'g--', 'LineWidth', 1, 'DisplayName', 'Funzione reale');
legend('Location', 'best');
title('Support Vector Regression con Level Bundle Method');
xlabel('X');
ylabel('Y');
grid on;
hold off;

% Stampa alcune metriche
mse = mean((sin(3*X_test) - Y_pred).^2);
fprintf('MSE sulla funzione originale: %.6f\n', mse);

% Evidenzia i support vector
figure;
hold on;
scatter(X, Y, 'ro', 'DisplayName', 'Dati');
scatter(X(sv_indices), Y(sv_indices), 100, 'bo', 'filled', 'DisplayName', 'Support Vectors');
plot(X_test, Y_pred, 'g-', 'LineWidth', 2, 'DisplayName', 'SVR con LBM');
legend('Location', 'best');
title('Support Vectors identificati');
xlabel('X');
ylabel('Y');
grid on;
hold off;