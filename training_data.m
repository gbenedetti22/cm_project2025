function [X,Y] = training_data(dataset)

switch lower(dataset)
    case "abalone"
        data = readtable('abalone.data', 'FileType', 'text', 'Delimiter', ',');
        data.Var1 = grp2idx(data.Var1);
        
        X = table2array(data(:, 1:end-1));
        Y = table2array(data(:, end));

    case "red_wine"
        data = readtable(fullfile('data', 'winequality_red.data'),...
                        'Delimiter', ';', 'FileType', 'text');
        
        X = table2array(data(:, 1:end-1));
        Y = table2array(data(:, end));

    case "white_wine"
        data = readtable(fullfile('data', 'winequality-white.data'),...
                        'Delimiter', ';', 'FileType', 'text');
        
        X = table2array(data(:, 1:end-1));
        Y = table2array(data(:, end));
    
    case "airfoil"
        data = readtable('airfoil.data', 'FileType', 'text', 'Delimiter', '\t');
        X = table2array(data(:, 1:end-1));
        Y = table2array(data(:, end));

    case "sin"
        X = linspace(-2, 2, 100)';
        Y = sin(3*X) + 0.1 * randn(size(X));
    case "exp"
        X = linspace(-2, 2, 100)';
        Y = exp(X) + 0.2*randn(size(X));
    case "poly"
        X = linspace(-3, 3, 100)';
        Y = 0.5*X.^3 - 2*X.^2 + X + 0.3*randn(size(X));
    case "step"
        X = linspace(-3, 3, 100)';
        Y = (X > 0) + 0.15*randn(size(X));
    case "outlier"
        X = linspace(-2, 2, 100)';
        Y = sin(X);
        outlier_idx = randperm(100, 10);  % 10 outlier casuali
        Y(outlier_idx) = Y(outlier_idx) + 2*randn(10,1);
    case "friedman"
        rng(98)

        n_samples = 500;
        n_features = 100;

        X = rand(n_samples, n_features);
        Y = 10 * sin(pi * X(:,1) .* X(:,2)) + 20 * (X(:,3) - 0.5).^2 + 10 * X(:,4) + 5 * X(:,5);
        Y = Y + randn(n_samples, 1);

        rng("default");
    otherwise
        error("No dataset found for: " + dataset)

end

end
