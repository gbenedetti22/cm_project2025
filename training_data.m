function [X,Y] = training_data(dataset)

switch lower(dataset)
    case "abalone"
        data = readtable('abalone.data', 'FileType', 'text', 'Delimiter', ',');
        data.Var1 = grp2idx(data.Var1);
        
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
        
    otherwise
        error("No dataset found for: " + dataset)

end

end