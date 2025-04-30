\newpage
# Conclusions: Achieved Results

In this chapter, we present the results obtained on various datasets, comparing the SVR with Level Bundle Method (LBM) against a classical SVR implementation, referred to as the *Oracle*. The hyperparameters used in each experiment are reported, and efforts were made to keep them as similar as possible to ensure a fair comparison.

As stated in the introductory chapter, the goal of this project is **not** to achieve the best possible MSE through extensive hyperparameter tuning. Instead, the objective is to demonstrate that the Level Bundle Method can deliver **comparable—if not superior—performance** to that of a classical SVR.

## Introduction

The dataset used are: [Abalone](https://archive.ics.uci.edu/dataset/1/abalone), [Airfoil](https://www.kaggle.com/datasets/fedesoriano/airfoil-selfnoise-dataset), [Red Whine](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) and Friedman function

The parameters for the Level Bundle Method used are:

- **tol**

  Tolerance for stopping criterion.

- **theta**

  Controls the trade-off between cutting plane approximation and descent direction.

- **lr** *(learning rate)*

  Step size for the gradient-based update.

- **momentum**

  Momentum term to accelerate convergence and avoid local minima.

- **scale_factor**

  Scaling coefficient applied to linear and quadratic terms.

- **max_constraints**

  Maximum number of cutting planes (constraints) maintained in the bundle.

\newpage

### Common Parameters

This parameters are used **on all dataset** (except where specified) by both SVR (Oracle and SVR with LBM).

| Parameter | Value          |
| :-------- | -------------- |
| Kernel    | RBF(sigma=0.5) |
| C         | 1              |
| Epsilon   | 0.05           |

\newpage

## Abalone

### Oracle
| Iterations | MSE    | Time (s) |
| ---------- | ------ | -------- |
| 60         | 4.2186 | 292.2166 |
| 90         | 4.2186 | 796.9473 |


### SVR with Level Bundle Method (LBM)

- **tol**: `1e-2` 
- **theta**: `0.5` 
- **lr** (learning rate): `1e-07` 
- **momentum**: `0.3` 
- **scale_factor**: `1e-05` 
- **max_constraints**: `60` 

| Iterations | MSE    | Time (s) |
| ---------- | ------ | -------- |
| 60         | 4.2819 | 28.2933  |
| 90         | 4.1963 | 88.0550  |

As observed, the *Oracle* achieves excellent results in just a few iterations, which suggests that `fmincon` is a highly effective solver for this specific type of problem. We also experimented with other alternatives, such as `quadprog`, but the resulting Hessian matrix proved to be unmanageable for large-scale problems like this one, making its use impractical.

An interesting observation is that once the minimum is reached, `fmincon` tends to plateau without further improvements. However, the major drawback is the training time: the average time between iterations is around **6 seconds**, resulting in very high total runtimes.

Our SVR implementation with LBM, on the other hand, performs slightly worse with the same number of iterations, but when increasing the iteration count moderately, it achieves an **even lower MSE** than the Oracle.

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{./assets/time.jpg}
\end{figure}

This graph shows the behavior of the function at the same time t, using the best parameters for both SVRs. Since the times for the 2 SVRs are markedly different, the values have been interpolated to the values at the common time, so it is easier to understand what happens at time t.

One particularly interesting aspect, evident from this time-based plot, is that up to around **60 seconds**, our LBM-SVR approaches the function minimum much faster, while the Oracle lags significantly behind. Notably, there exists a time `t` at which both models reach **similar performance levels**.

![Relative gap between the Oracle and LBM function evaluations](./assets/cazzillo_2.jpg)

Another key observation comes from the **relative gap** plot: despite the differences in runtime, the learning behavior is remarkably comparable. This indicates that the introduction of regularization had a **stabilizing effect** on the optimization dynamics.

\newpage

## Airfoil

| Parameter | Value          |
| :-------- | -------------- |
| Kernel    | RBF(sigma=0.6) |

### Oracle

| Iterations | MSE     | Time (s) |
| ---------- | ------- | -------- |
| 60         | 10.7398 | 54.05    |
| 90         | 10.7397 | 94.32    |

### SVR with Level Bundle Method (LBM)

- **tol**: `1e-2` 
- **theta**: `0.57 
- **lr** (learning rate): `1e-07` 
- **momentum**: `0.34 
- **scale_factor**: `4.6416e-05` 
- **max_constraints**: `60` 

| Iterations | MSE     | Time (s) |
| ---------- | ------- | -------- |
| 60         | 10.7976 | 19.13    |
| 90         | 10.7354 | 59.04    |

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{./assets/airfoil_90.jpg}
\end{figure}

Unlike the Abalone dataset results, here the two SVR models show a highly synchronized learning behavior, supported by the minimal discrepancy in their MSE values

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{./assets/time_airfoil.jpg}
\end{figure}

The time-dependent behavior is noteworthy: while the function evaluation shows a substantial initial discrepancy, the difference almost completely disappears after 10 seconds, resulting in virtually indistinguishable outputs.

\newpage

## Red Whine

### Oracle

| Iterations | MSE      | Time (s) |
| ---------- | -------- | -------- |
| 60         | 0.055518 | 33.87    |
| 90         | 0.055519 | 59.08    |

### SVR with Level Bundle Method (LBM)

- **tol**: `1e-2` 
- **theta**: `0.4878`
- **lr** (learning rate): `4.1169e-06` 
- **momentum**: `0.654`
- **scale_factor**: `3.7483e-05` 
- **max_constraints**: `60` 

| Iterations | MSE      | Time (s) |
| ---------- | -------- | -------- |
| 60         | 0.057641 | 26.29    |
| 90         | 0.055734 | 51.52    |

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{./assets/red_wine.jpg}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{./assets/winered_time.jpg}
\end{figure}

As with the Red Wine dataset, here too we observe no significant differences between the two SVR models: both exhibit excellent and comparable convergence, evidenced by the small MSE gap and similar computational time.

\newpage

## Summary

| Dataset   | Metodo    | Iterazioni | MSE      | Tempo (s) |
| --------- | --------- | ---------- | -------- | --------- |
| Abalone   | Oracle    | 60         | 4.2186   | 292.2166  |
| Abalone   | LBM (SVR) | 60         | 4.2819   | 28.2933   |
| Abalone   | Oracle    | 90         | 4.2186   | 796.9473  |
| Abalone   | LBM (SVR) | 90         | 4.1963   | 88.0550   |
| Airfoil   | Oracle    | 60         | 10.7398  | 54.05     |
| Airfoil   | LBM (SVR) | 60         | 10.7976  | 19.13     |
| Airfoil   | Oracle    | 90         | 10.7397  | 94.32     |
| Airfoil   | LBM (SVR) | 90         | 10.7354  | 59.04     |
| Red Whine | Oracle    | 60         | 0.055518 | 33.87     |
| Red Whine | LBM (SVR) | 60         | 0.057641 | 26.29     |
| Red Whine | Oracle    | 90         | 0.055519 | 59.08     |
| Red Whine | LBM (SVR) | 90         | 0.055734 | 51.52     |

