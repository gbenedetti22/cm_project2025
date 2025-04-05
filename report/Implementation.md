\newpage

# Implementation

## Dataset

Initially, we used simple functions such as sine, exponential, and step functions with added noise. These synthetic datasets allowed us to test and verify the correctness of our SVR implementation in terms of predictions and performance. <br/>
Subsequently, we moved to a larger and more complex dataset: the **Abalone dataset**. This is a regression dataset where the goal is to predict the age of abalones based on various features.

The dataset is structured as follows:

- The first 8 columns correspond to the input features.
- The last column represents the target value (age of the abalone).

Firstly, we define the SVR dual function, which computes both the function value `f` and the corresponding subgradient `g`:

```matlab
Input:
    x			% input vector
    y			% target vector
    epsilon		% scalar value
    K           % kernel matrix

Output:
	f			% objective function value
	g			% gradient
    
f = 0.5 * TRANSPOSE(x) * (K * x) + epsilon * SUM(ABS(x)) - TRANSPOSE(y) * x
g = K * x + epsilon * SIGN(x) - y
```

Before training, the features are normalized.

\newpage

## (A2) SVR general-purpose solver 

For the implementation of a generic SVR solver, we formulated the dual problem using MATLAB’s lambda functions. Unlike quadprog, fmincon does not require explicit Hessian matrix definitions—a key advantage that significantly improves scalability for large-scale problems

```matlab
% Starting point
alpha0 = zeros(n, 1);

% Inequality constraints
A = [];
b = 0;

% Equality constraint: sum(alpha) = 0
Aeq = ones(1, n);
beq = 0;

% Lower and upper bounds: 0 <= alpha <= C
lb = -C * ones(n, 1);
ub = C * ones(n, 1);
```

 Once the execution is completed, we extract the **support vectors** from the solution.

```matlab
% Identify indices of support vectors
% Select indices where the absolute value of alpha is greater than a threshold (tol)
sv_indices = indices where |alpha[i]| > tol

% Compute the bias term for prediction
bias = mean( Y[i] - sum over j of K[i][j] * alpha[j] ), for all i in sv_indices
```

### Performance Evaluation

We evaluated this SVR implementation on both synthetic datasets and the **Abalone dataset**. While the model generalizes well on synthetic data, its performance deteriorates significantly on the Abalone dataset due to the **high number of constraints**. 

#### Synthetic data

- **Sine function**
  $$ Y = \sin(3X) + 0.1 \cdot \mathcal{N}(0, 1). $$
  ![](./assets/sin.jpg)
- **Outliers test**
  $$ Y = \sin(X) + \underbrace{2 \cdot \mathcal{N}(0, 1)}_{\text{10 random points}}. $$
  ![](./assets/outlier.jpg)

\newpage

#### Abalone

- **MSE**: 4.2187

![](./assets/fmincon_plot.jpg)

\newpage

## (A1) Level Bundle Method Implementation

Previously, we mathematically formulated the objective function and constraints of the **Level Bundle Method (LBM)** in a format compatible with `quadprog`. We now translate these formulations into code, ensuring a direct correspondence between the mathematical expressions and their implementation. Firstly we define the solver signature function as:

```matlab
alpha_opt = mp_solve(alpha_hat, bundle, f_level, C)
```

Where:

* `alpha_hat` is the current $\alpha$ 
* `bundle` is a structure containing:
  * **bundle.alpha** $\to$  vector of dual variables $\alpha$
  * **bundle.g** $\to$ vector of the subgradients
  * **bundle.f** $\to$ vector of function evaluations
* `f_level` is the current level

### Hessian Matrix and Linear Coefficients

The objective function of the optimization problem is defined as:
$$
H = I, \quad f = -\hat{x}_k.
$$

In pseudo-code, this is implemented as:

```matlab
H = blkdiag(eye(n), 0); % n = length(alpha_hat)
f = [-alpha_hat; 0];
```

### Inequality Constraints

The linear constraints are represented in matrix form as:
$$
A = \begin{bmatrix} \xi_k^\top & -1 \end{bmatrix}, \quad b = \langle \xi_k, \hat{\alpha_k} \rangle - f(\hat{\alpha_k})
$$

Translated into MATLAB:

```matlab
A = [bundle.g' -ones(m, 1)]; % m = length(bundle.f)
b = sum(bundle.g .* bundle.alpha, 1)' - bundle.f';
```

### Equality Constraints

The equality constraints are given by:
$$
A_{\text{eq}} = \begin{bmatrix} \mathbf{1}^\top \end{bmatrix}, \quad b_{\text{eq}} = 0.
$$

MATLAB implementation:

```matlab
Aeq = [ones(1, n) 0];
beq = 0;
```

### Variable Bounds

The variables are subject to the following bounds:
$$
\begin{aligned}
lb &= [-C,\, -\infty], \\
ub &= [C,\, f_{\text{level}}].
\end{aligned}
$$

Implemented as:

```matlab
lb = [-C * ones(n, 1); -inf];
ub = [C * ones(n, 1); f_level];
```

\newpage

### Solving the Optimization Problem

Finally, we use `quadprog` to find the optimal solution. So the ending function is:

```matlab
function alpha_opt = mp_solve(alpha_hat, bundle, f_level, C)
    n = length(alpha_hat);
    m = length(bundle.f);

    H = blkdiag(eye(n), 0);
    f = [-alpha_hat; 0];

    A = [bundle.g' -ones(m, 1)];
    b = sum(bundle.g .* bundle.alpha, 1)' - bundle.f';

    Aeq = [ones(1, n) 0];
    beq = 0;

    lb = [-C * ones(n, 1); -inf];
    ub = [C * ones(n, 1); f_level];

    options = optimoptions('quadprog','Display','off');

    [sol, ~, exitflag] = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], options);

    if exitflag <= 0
        warning("best solution not found, keeping prev...");
        alpha_opt = alpha_hat;
    else
        alpha_opt = sol(1:n);
    end
end
```


\newpage

## **LBM Algorithm: Pseudo-code Implementation**

```matlab
Input:
    K           % kernel matrix
    alpha       % current solution (dual vector)
    C           % upper bound on dual variables
    theta       % level parameter
    tol         % convergence tolerance
    max_iter    % maximum number of iterations

% Initialization
[f, g] = svr_dual_function(alpha)
f_best = f

bundle.alpha = alpha
bundle.f = f
bundle.g = g

for iter = 1 to max_iter:
    % Compute the acceptance level
    level = theta * f + (1 - theta) * f_best

    % Solve the master problem to get the new point
    alpha_new = mp_solve(alpha, bundle, level, C)

    % Evaluate the function and subgradient at alpha_new
    [f_new, g_new] = svr_dual_function(alpha_new, K, y, epsilon)

    % Check if the new point is acceptable
    if f_new < f_best:
        f_best = f_new
    
    if f_new <= level:
        alpha = alpha_new
        f = f_new
    
    % Update bundle
    bundle.alpha = [bundle.alpha, alpha_new]
    bundle.f = [bundle.f, f_new]
    bundle.g = [bundle.g, g_new]

    % Check for convergence
    if ||alpha_new - alpha|| < tol:
        break

return alpha  % Return the optimal solution
```

---

# Performance Evaluation

## SVR Training with LBM on Synthetic Data

To evaluate the performance of our SVR (Support Vector Regression) model implemented with the Level Bundle Method (LBM), we conducted preliminary testing on predefined synthetic datasets. The results demonstrate that the model retains the generalization capability characteristic of classical SVR while maintaining comparable computational efficiency in terms of execution time.

Below, we present the predictive performance on the sine function as a representative example of the model’s generalization capabilities. The remaining functions (omitted for brevity) demonstrate behaviors fully aligned with the standard SVR implementation.

- **Sine function**
  $$Y = \sin(3X) + 0.1 \cdot \mathcal{N}(0, 1). $$
  ![](./assets/sin_lbm.jpg)

### Training SVR with LBM on the Abalone Dataset

As mentioned earlier, we used the Abalone dataset to evaluate the actual performance of our SVR implementation. This dataset contains 4177 samples with 8 features each, making it a perfect candidate as it introduces challenges such as:

- A high number of constraints
- Increased memory consumption
- Prolonged training time due to repeated calls to the solver

#### High Number of Constraints

The main issue encountered in the implementation of the **Level bundle method** in MATLAB is the uncontrolled growth of the bundle size at each iteration.

Every cycle of the algorithm adds new constraints to the system, leading to an **exponential increase in the number of conditions** to be handled. This progressive expansion of the bundle has two critical effects:

1. **Memory overload**: The bundle data structure, especially with high-dimensional datasets, consumes resources in a drastically non-linear manner, making allocation unsustainable for large data volumes.
2. **Performance degradation of quadprog**: MATLAB’s quadratic programming solver becomes progressively slower—sometimes even leading to total stalls—due to the need to process constraint matrices with thousands of rows.

To address this issue, we introduced the **bundle truncation** technique: once a certain threshold is exceeded, the oldest constraints are removed from the bundle, leaving only the most recent $k$ constraints. This significantly improved both memory efficiency and computational performance.

```matlab
if size(bundle.alpha, 2) > max_constraints
    bundle.alpha = bundle.alpha(:, 2:end);
    bundle.f = bundle.f(2:end);
    bundle.g = bundle.g(:, 2:end);
end
```

#### Memory Consumption

Despite the introduction of bundle truncation, memory consumption remained significantly high. This was mainly due to the allocation of large matrices. For instance, given that the dataset contains 4177 features, the matrix $H$ passed to `quadprog` has a size of $4178 \times 4178$, which is excessive for an identity matrix.

To mitigate this issue, we decided to switch to **sparse matrices**, which drastically reduced memory usage, as only nonzero elements are stored. Additionally, this optimization ensured that the `quadprog` solver could terminate within a reasonable time frame. So the new `mp_solve` function can be written as:

```matlab
H = blkdiag(speye(n), 0);
f = sparse([-alpha_hat; 0]);

A = sparse([bundle.g' -ones(m, 1)]);
b = sparse(sum(bundle.g .* bundle.alpha, 1)' - bundle.f');

Aeq = sparse([ones(1, n) 0]);
beq = 0;

lb = sparse([-C * ones(n, 1); -inf]);
ub = sparse([C * ones(n, 1); f_level]);
```

#### Performance Optimization: Convergency

At this stage, our SVR implementation using the Level Bundle Method (LBM) successfully completes in reasonable time and achieves an **MSE** of **4.3729**, which is very close to the Oracle's performance, thanks to the optimizations implemented thus far. However, the convergence <u>remains relatively slow</u>, requiring more than **200 iterations**, unlike the Oracle which approaches the minimum after approximately **40 iterations**.

This phenomenon is primarily due to the step sizes the algorithm takes: while the Oracle maintains an average step size of approximately 98, our algorithm averages only 1e-1. This limitation is largely attributable to the **matrix H** and the **linear term** that inherently penalize larger steps, thereby slowing convergence.

To address this issue, we introduced a new hyperparameter called "*scale_factor*" that modifies our objective function:
$$
\frac{1}{2} \cdot \text{scale\_factor} \cdot \alpha^T\alpha - \text{scale\_factor} \cdot \alpha_k^T\alpha
$$
This scaling essentially **reduces the quadratic penalty** while proportionally adjusting the linear term, effectively increasing the **trust region** and consequently allowing larger steps. This modification enabled us to increase average step sizes from *1e-1* to approximately *87*, resulting in an improved **MSE** of **4.2569**, which is comparable to our **Oracle's performance (4.2191).**

Nevertheless, despite the scaling factor, the algorithm still tends to decelerate significantly as it approaches the minimum value. In fact, to achieve that MSE value, we needed to further increase the number of iterations (approximately **120-150**). To overcome this limitation, we introduced **momentum** with **learning rate** applied to the alpha variables:
$$
v_\text{k+1} = \beta  v_k - \eta \cdot \xi_k \\
\alpha_\text{k+1} = \alpha_k + v_\text{k+1}
$$
Where:

- $\beta$ is the momentum factor
- $v_k$ is the velocity
- $\eta$ is the learning rate
- $\xi$ is the sungradient

In pseudo-code:

```matlab
velocity = momentum * velocity - learning_rate * g;
alpha = alpha + velocity;

---

H = blkdiag(speye(n) * scale_factor, 0);
f = sparse([-alpha_hat * scale_factor; 0]);
```

This implementation ensures that the method initially takes **large steps** and gradually slows down as it approaches the optimum, while still maintaining adequate step sizes. Unfortunately, this introduced two new hyperparameters (learning rate and momentum) which, if not appropriately selected, can cause the objective function to **diverge**.

#### Choosing the Solver: Is Quadprog Really the Best Option?

The current implementation relies on **quadprog**, with a training time of approximately **50 seconds**, but alternative solvers could significantly improve time efficiency.

Further gains can be achieved by adopting **high-performance solvers** optimized for large-scale problems, as such alternatives leverage best algorithms and parallelization. This could reduce training times by **an order of magnitude** without compromising accuracy.