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
        
    otherwise
        error("No dataset found for: " + dataset)

end

end