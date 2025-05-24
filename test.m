clc; clear; close all;

% Available: abalone, white_wine, red_wine, airfoil
dataset = "airfoil";

% Load and normalize the selected dataset
[X, y] = training_data(dataset);
X = zscore(X);

% Load optimal parameters
[svr_params_lbm, svr_params_oracle] = get_params(dataset);
svr = SVR(svr_params_oracle);
svr_lbm = SVR(svr_params_lbm);

% Fit the training data
[x, h] = svr.fit(X, y);
[x_lbm, h_lbm] = svr_lbm.fit(X, y);

% Compare performance of both models
analyze_results(X, y, svr, svr_lbm, h, h_lbm);
