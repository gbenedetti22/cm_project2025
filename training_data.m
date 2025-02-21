function [X,Y] = training_data(dataset)

switch lower(dataset)
    case "abalone"
        data = readtable('abalone.data', 'FileType', 'text', 'Delimiter', ',');
        data.Var1 = grp2idx(data.Var1);
        
        X = table2array(data(:, 1:end-1));
        Y = table2array(data(:, end));
    otherwise
        error("No dataset found for: " + dataset)

end

end