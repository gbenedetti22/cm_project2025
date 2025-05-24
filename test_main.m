%% Project 10 - SVR with Level Bundle Method
% -----------------------------------
% This project implements and compares two SVR approaches:
% 1) ORACLE: Standard SVR with theoretically optimal parameters
% 2) SVR + LBM: SVR with parameters optimized via Level Bundle Method
%
% Key Outputs:
% - MSE: Oracle and LBM
% - Relative Gap: Difference between best minimum found by Oracle and LBM
% - Computation Time
% - Relative gap across iterations and objective values over time
%
% Usage: Simply select the dataset to evaluate different benchmarks.
% Results are automatically analyzed and displayed.

clc; clear; close all;

% Available: abalone, white_wine, red_wine, airfoil
dataset = "airfoil";

% Load and normalize the selected dataset
[X, y] = training_data(dataset);
X = zscore(X);

% Load optimal parameters
[lbm_params, oracle_params] = get_params(dataset);
svr = SVR(oracle_params);
svr_lbm = SVR(lbm_params);

% Fit the training data
[x, h] = svr.fit(X, y);
[x_lbm, h_lbm] = svr_lbm.fit(X, y);

% Compare performance of both models
analyze_results(X, y, svr, svr_lbm, h, h_lbm);