clc; clear; close all;

[X, y] = training_data("abalone");
X = zscore(X);

lbm_params = struct(...
    'max_iter',       200, ...    % Maximum iterations
    'epsilon',        0.05, ...   % Convergence tolerance
    'tol',            1e-11, ...    % Subgradient step tolerance
    'theta',          0.9, ...    % Convex combination parameter
    'max_constraints', 50, ...    % Max constraints in bundle
    'qp_ratio',       0 ...       % 0 = always subgradient method
); 

lbm = LBM(lbm_params);

svr_params = struct(...
    'kernel_function', RBFKernel(0.7), ...  % Kernel function
    'C',              1, ...             % Regolarization parameter
    'epsilon',        0.05, ...           % Epsilon margin
    'opt',            lbm ...            % LBM optimizer
); 
svr = SVR(svr_params);


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
% scatter(X_sv, Y_sv, 'ks', 'MarkerFaceColor', 'g', 'DisplayName', 'Support Vectors');
% xlabel('X'); ylabel('y'); title('SVR Duale (RBF) con Bundle Method');
% legend; grid on;
