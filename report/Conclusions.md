\newpage
# Conclusions: Achieved Results

This chapter presents a comparative analysis between an SVR using the Level Bundle Method (LBM) and a classical SVR implementation, referred to as the *Oracle*. Experiments were conducted across diverse datasets, with similar hyperparameters to ensure fairness. The goal is to evaluate under which conditions Level Bundle Method (LBM), achieves solution quality and computational performance comparable to the Oracle.

Oracle solution validity was verified using MOSEK logs, focusing on solution status (e.g., OPTIMAL, NEAR_OPTIMAL or UNKNOWN) and the duality gap (considered reliable if below 1e-4). 

<u>In general, we consider the comparison as a win if the relative gap, between Oracle and SVR with Level Bundle Method (LBM), is less than 1e-4.</u>

## Introduction

The dataset used are: [Abalone](https://archive.ics.uci.edu/dataset/1/abalone), [White Whine](https://www.kaggle.com/datasets/piyushagni5/white-wine-quality), [Red Whine](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) and [Airfoil](https://www.kaggle.com/datasets/fedesoriano/airfoil-selfnoise-dataset) :

| Dataset    | Inputs  | Features |
| ---------- | ------- | -------- |
| Abalone    | 4,177   | 8  |
| White Wine | 4,898   | 11 |
| Red Wine   | 1,599   | 11 |
| Airfoil    | 1,503   | 5  |

The parameters for the Level Bundle Method used are:

- **tol**

  Tolerance for stopping criterion, indicating the distance between the lower and upper bound of the Level Bundle Method.

- **theta**

   Controls the step balance between the current lower bound  and the best-known solution.

- **max_constraints**

  Maximum number of cutting planes (constraints) maintained in the bundle.



Each dataset is analyzed individually, reporting the parameters used for both the Oracle and Level Bundle Method (LBM), and presenting two key plots:

- **Relative Gap Plot**:
  - *Y-axis*: Relative gap between Oracle and LBM (logarithmic scale)
  - *X-axis*: Normalized number of iterations
  - *Purpose*: To assess how closely the LBM-based SVR approaches the Oracle’s solution at each normalized iteration.
- **Convergence Plot**:
  - *Y-axis*: Objective values
  - *X-axis*: Time in seconds
  - *Purpose*: To evaluate how quickly the LBM-based SVR progresses toward the Oracle’s solution over time (function evaluation at time $t$)

\newpage

## Abalone

| Parameter | Value          |
| :-------- | -------------- |
| Kernel    | RBF(sigma=0.4) |
| C         | 1              |
| Epsilon   | 1e-6        |

### Oracle
| Iterations | MSE    | Min        | Time (s) |
| ---------- | ------ | ---------- | -------- |
| 14         | 4.2092 | -6087.4994 | 27.4832  |

```toml
Interior-point solution summary
  Problem status  : PRIMAL_AND_DUAL_FEASIBLE
  Solution status : OPTIMAL
  Primal.  obj: -6.0874994035e+03   nrm: 1e+00    Viol.  con: 6e-09    var: 0e+00  
  Dual.    obj: -6.0874994815e+03   nrm: 2e+01    Viol.  con: 0e+00    var: 8e-06   
  Gap primal-dual: 1.2813e-08
```
### SVR with Level Bundle Method (LBM)

- **tol**: `2e-5` 
- **theta**: `0.6`   
- **max_constraints**: `100` 

| Iterations | MSE    | Min        | Time (s) |
| ---------- | ------ | ---------- | -------- |
| 86         | 4.2122 | -6087.4778 | 34.6167  |

#### Relative Gap

| Lower-Upper Bound | Oracle-LBM (Minimum) |
| ---------------------	| -------------- |
| 1.973186e-05 | 3.5475e-06 |

\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/abalone_gap.jpg}
\end{figure}

The plot shows a clear decreasing trend in the relative gap between the Oracle and LBM methods across normalized iterations. Although LBM requires a significantly higher number of iterations (86) compared to the Oracle (14), the normalized view allows for a direct comparison of their progression.

The gap reduces steadily overall, with a noticeable drop in the middle and final stages. Interestingly, there is a brief increase in the gap before the final convergence, indicating a temporary divergence before the LBM method ultimately reaches the minimum.

\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/abalone_time.jpg}
\end{figure}

Observing the performance over time, it is evident that the LBM method exhibits a much steeper initial decrease in the objective value compared to the Oracle. However, after reaching a plateau, it requires more iterations to converge fully to the minimum.

The Oracle, on the other hand, reaches the minimum slightly faster in terms of total runtime (27.5 seconds versus 34.6 seconds for LBM). Nevertheless, the difference in runtime is relatively small, and the additional overhead of preserving bundle information appears to have limited impact on performance in this dataset, which is characterized by a large number of inputs and features.

\newpage

## White Wine

| Parameter | Value          |
| :-------- | -------------- |
| Kernel    | RBF(sigma=0.6) |
| C         | 1              |
| Epsilon   | 1e-6           |

### Oracle
| Iterations | MSE      | Min | Time (s)  |
| ---------- | -------- | --------- | --------- |
| 12        | 0.068958 | -1257.6789 | 28.9418 |

```toml
Interior-point solution summary
  Problem status  : PRIMAL_AND_DUAL_FEASIBLE
  Solution status : OPTIMAL
  Primal.  obj: -1.2576789125e+03   nrm: 1e+00    Viol.  con: 1e-10    var: 0e+00  
  Dual.    obj: -1.2576789122e+03   nrm: 6e+00    Viol.  con: 0e+00    var: 2e-07  
  Gap primal-dual: 2.3853e-10
```



### SVR with Level Bundle Method (LBM)

- **tol**: `2e-5` 
- **theta**: `0.7`   
- **max_constraints**: `100` 

| Iterations | MSE      | Min   | Time (s)  |
| ---------- | -------- | --------- | --------- |
| 114     | 0.068975 | -1257.6693 | 63.3961 |

#### Relative Gap

| Lower-Upper Bound | Oracle-LBM (Minimum) |
| ---------------------	| -------------- |
| 1.946468e-05 | 6.8581e-06 |


\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/white_wine_gap.jpg}
\end{figure}

The convergence of both methods over the iterations shows that they reach a plateau in the final stages, indicating a more stable — though still overall decreasing — trend toward the minimum, with progressively smaller differences in the obtained results.

\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/white_wine_time.jpg}
\end{figure}

As observed in the Abalone dataset, the Oracle exhibits a less steep decrease in terms of both runtime and objective value. However, the difference between the two methods is significant, with LBM showing approximately a 50% increase in runtime — from 28.9 to 63.4 seconds. 



\newpage

## Red Wine

| Parameter | Value           |
| :-------- | --------------- |
| Kernel    | RBF(sigma=0.55) |
| C         | 1               |
| Epsilon   | 1e-6            |

### Oracle
| Iterations | MSE      | Min | Time (s)  |
| ---------- | -------- | --------- | --------- |
| 12     | 0.056638 | -380.0152 | 2.6541 |

```toml
Interior-point solution summary
  Problem status  : PRIMAL_AND_DUAL_FEASIBLE
  Solution status : OPTIMAL
  Primal.  obj: -3.8001520940e+02   nrm: 1e+00    Viol.  con: 6e-11    var: 0e+00  
  Dual.    obj: -3.8001514184e+02   nrm: 6e+00    Viol.  con: 0e+00    var: 3e-05  
  Gap primal-dual: 1.7778e-07
```
### SVR with Level Bundle Method (LBM)

- **tol**: `1e-6` 
- **theta**: `0.7`   
- **max_constraints**: `100` 

| Iterations | MSE    | Min | Time (s) |
| ---------- | ------ | -------- | -------- |
| 170  | 0.056647 | -380.0144 | 61.7583 |

#### Relative Gap

| Lower-Upper Bound | Oracle-LBM (Minimum) |
| ---------------------	| -------------- |
| 9.820069e-07 | 2.1485e-06 |

\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/wine_gap.jpg}
\end{figure}

Similar to the White Wine dataset, both methods reach a plateau in the final stages, reducing the relative gap between the minimum values achieved. Notably, in this case, the number of iterations is significantly higher: the Oracle converges within just 12 iterations, while the LBM requires 170 iterations to reach comparable results.

\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/wine_time.jpg}
\end{figure}

In terms of time-based performance, a significant difference is observed. Although LBM exhibits a faster initial phase (consistent with prior studies), its total computation time (61.8 seconds) is significantly higher compared to the Oracle (2.6 seconds). This discrepancy is attributed to LBM's larger number of iterations and the overhead of bundle storage.

\newpage

## Airfoil

| Parameter | Value          |
| :-------- | -------------- |
| Kernel    | RBF(sigma=0.7) |
| C         | 1              |
| Epsilon   | 1e-7           |

### Oracle

| Iterations | MSE      | Min | Time (s)  |
| ---------- | -------- | --------- | --------- |
| 13       | 10.24 | -4635.7585 | 1.6597 |

```toml
Interior-point solution summary
  Problem status  : PRIMAL_AND_DUAL_FEASIBLE
  Solution status : OPTIMAL
  Primal.  obj: -4.6357585385e+03   nrm: 1e+00    Viol.  con: 1e-10    var: 1e-07  
  Dual.    obj: -4.6357585894e+03   nrm: 1e+02    Viol.  con: 0e+00    var: 8e-06
  Gap primal-dual: 1.0980e-08
```
### SVR with Level Bundle Method (LBM)

- **tol**: `1e-6` 
- **theta**: `0.6`   
- **max_constraints**: `100` 

| Iterations | MSE      | Min | Time (s)  |
| ---------- | -------- | --------- | --------- |
| 88     | 10.241 | -4635.7580 | 14.6318 |

#### Relative Gap

| Lower-Upper Bound | Oracle-LBM (Minimum) |
| ---------------------	| -------------- |
| 9.674982e-07 | 1.1165e-07 |

\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/airfoil_gap.jpg}
\end{figure}

For the airfoil dataset, the relative gap plot highlights a more consistent descent, reaching a minimal gap between the models and yielding the best overall result compared to the other datasets in the final stages. The disparity in the number of iterations is still notable, with the oracle converging using only about 1/8 the iterations.

\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/airfoil_time.jpg}
\end{figure}

In terms of time, the overall behavior shown in this graph is consistent with previous findings; however, on this dataset, LBM exhibits a less steep initial decrease, which can be attributed to the algorithm taking larger steps in the early phase. The time difference is also less significant than for the Redwine dataset (which is the most similar case among those studied), with the oracle requiring 1.7 seconds compared to LBM, which takes only an additional 12 seconds.

## Sublinear Convergence

\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/abalone_conv.jpg}
\end{figure}

As discussed in the theoretical section of the report, the algorithm is expected to exhibit sublinear convergence. This behavior is clearly reflected in the graph, where the gradual reduction in the gap over many iterations results in the characteristic curvature typical of sublinear convergence patterns.

\newpage

## Summary

| Dataset     | Model | Iterations | MSE      | Min   | Time (s) |
| ----------- | --------- | ---------- | -------- | --------- | --------- |
| Abalone     | Oracle    | 14         | 4.2092   | -6087.499  | 27.4832   |
| Abalone     | LBM | 86         | 4.2122   | -6087.477  | 34.6167   |
| White Wine | Oracle    | 12         | 0.068958 | -1257.678  | 28.9418   |
| White Wine | LBM | 114        | 0.06897 | -1257.669  | 63.3961   |
| Red Wine   | Oracle    | 12         | 0.056638 | -380.0152   | 2.6541    |
| Red Wine   | LBM | 170        | 0.056647 | -380.0144   | 61.7583   |
| Airfoil     | Oracle    | 13         | 10.24    | -4635.758  | 1.6597    |
| Airfoil     | LBM | 88         | 10.241   | --4635.7580 | 14.6318   |

#### Relative Gap

| Dataset    | Lower-Upper Bound | Oracle-LBM |
| ---------- | -------------  | -------- |
| Abalone    | 1.973186e-05          | 3.5475e-06                     |
| White Wine | 1.946468e-05          | 6.8581e-06                     |
| Red Wine   | 9.820069e-07          | 2.1485e-06                     |
| Airfoil    | 9.674982e-07          | 1.1165e-07                     |

Across all datasets, the LBM-based SVR generally achieves comparable MSE and converges to the same minimum as the Oracle. However, computational time varies depending on dataset size and dimensionality.

- For **large, high-dimensional datasets** (e.g., *Abalone*, *White Wine*):
  - The time difference is small, as both methods require substantial computation.
  - LBM's overhead is less impactful, and convergence is similar.
- For **small or low-dimensional datasets** (e.g., *Airfoil*, *Red Wine*):
  - The Oracle is significantly faster (e.g., 1.7s vs. 14.7s for Airfoil, 2.6s vs. 61.8s for Red Wine).
  - LBM's overhead dominates, leading to notable inefficiency.

This shows that LBM is more competitive on complex problems, while its cost is harder to justify on simpler datasets.
