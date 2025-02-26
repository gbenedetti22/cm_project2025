function x_proj = project_onto_level(B, f_level)
    x_k = B{end}{1};  % Ultimo punto del bundle
    
    A = []; b = [];
    
    for i = 1:length(B)
        x_i = B{i}{1};
        f_i = B{i}{2};
        g_i = B{i}{3};
        
        A = [A; g_i'];
        b = [b; f_level - f_i + g_i' * x_i];
    end

    if isempty(A)
        x_proj = x_k;
        return;
    end

    n = length(x_k);
    H = eye(n);
    f = -x_k;

    options = optimoptions('quadprog', 'Display', 'off');
    [x_proj, ~, exitflag] = quadprog(2*H, f, A, b, [], [], [], [], [], options);

    if exitflag <= 0
        warning('quadprog non ha trovato una soluzione valida. Manteniamo x_k.');
        x_proj = x_k;
    end
end