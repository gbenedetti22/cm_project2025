clc; clear; close all;

% Carica i dati e normalizza le features
[X, y] = training_data("abalone");
X = zscore(X);

% Parametri per il Level Bundle Method
lbm_params = struct(...
    'max_iter',       500, ...    % Numero massimo di iterazioni
    'epsilon',        1e-6, ...   % Tolleranza per la convergenza
    'tol',            0.1, ...    % Tolleranza per il passo del subgradiente
    'theta',          0.5, ...    % Parametro per la combinazione convessa
    'max_constraints', 20, ...     % Numero massimo di cutting planes nel bundle
    'qp_ratio',       100 ...       % Se 100, usa sempre quadprog
); 
lbm = LBM(lbm_params);

% Parametri per il modello SVR
svr_params = struct(...
    'kernel_function', RBFKernel(), ...  % Funzione kernel (RBF)
    'C',              1, ...              % Parametro di regolarizzazione
    'epsilon',        0.1, ...            % Margine epsilon
    'opt',            lbm, ...            % Ottimizzatore: oggetto LBM
    'tol',            1e-5 ...            % Tolleranza per identificare i support vector
); 
svr = SVR(svr_params);

fprintf("Training start..\n");
tic
[X_sv, Y_sv] = svr.fit(X, y);
fprintf("Training end! :)\n");
toc

y_pred = svr.predict(X);

fprintf("MSE: %f\n", mse(y_pred, y));

%figure; hold on;
%plot(X, y, '-', 'LineWidth', 1, 'DisplayName', 'Dati training');
%plot(X, y_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'Predizione SVR');
%scatter(X_sv, Y_sv, 'ks', 'MarkerFaceColor', 'g', 'DisplayName', 'Support Vectors');
%xlabel('X'); ylabel('y'); title('SVR Duale (RBF) con Bundle Method');
%legend; grid on;
