\newpage
# Conclusions: Achieved Results

In this chapter, we present the results obtained on various datasets, comparing the SVR with Level Bundle Method (LBM) against a classical SVR implementation, referred to as the *Oracle*. The hyperparameters used in each experiment are reported, and efforts were made to keep them as similar as possible to ensure a fair comparison.

As stated in the introductory chapter, the goal of this project is **not** to achieve the best possible MSE through extensive hyperparameter tuning. Instead, the objective is to demonstrate that the Level Bundle Method can deliver **comparable—if not superior—performance** to that of a classical SVR.

**For each dataset, two plots will be displayed:**

1. **Relative Gap Plot**: This shows the relative gap between the oracle and the SVR with LBM (Level Bundle Method), calculated using the formula:
   $$
   \text{rg} = \frac{|f_k - f^*|}{|f^*|}  
   $$
   
2. **Function Value Plot**: This displays the function value at the same time *t*.

Since the two models have different training times, an interpolation between their respective data points was performed using MATLAB’s `interp1` function to align their results for comparison.

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

  Tolerance for stopping criterion.

- **theta**

   Controls the step balance between the current lower bound  and the best-known solution.

- **max_constraints**

  Maximum number of cutting planes (constraints) maintained in the bundle.

\newpage

## Abalone

| Parameter | Value          |
| :-------- | -------------- |
| Kernel    | RBF(sigma=0.4) |
| C         | 1              |
| Epsilon   | 0.1           |

### Oracle
| Iterations | MSE    | Time (s) |
| ---------- | ------ | -------- |
| 60         | 4.1976 | 334.2511 |

### SVR with Level Bundle Method (LBM)

- **tol**: `1e-2` 
- **theta**: `0.6`   
- **max_constraints**: `60` 

| Iterations | MSE    | Time (s) |
| ---------- | ------ | -------- |
| 60         | 4.1966 | 148.3282 |

\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/abalone_gap.jpg}
\end{figure}

The plot highlights a clear divergence in the early stages of the iterations. During the first 5–10 iterations, the relative gap for the Oracle exhibits a steep increase indicating a fast convergence to a near-optimal state in a few steps.

In contrast, the Level Bundle Method (LBM) begins with a noticeably higher relative gap and shows a more gradual increase over the initial iterations, characterized by smaller steps. Over the subsequent 15–20 iterations, the performance of both methods starts to align in terms of the magnitude of the relative gap and the step length of its decrease. Eventually, both algorithms reach a plateau where further reductions in the relative gap become minimal, suggesting they have both converged closely to their respective optimal function value, with no further noticeable differences in their convergence behavior.

\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/abalone_time.jpg}
\end{figure}

Observing the temporal performance within the first 50 seconds, the SVR with LBM achieves a considerably faster value reduction compared to the Oracle. 

The LBM reaches the value of approximately -5000 in about 10 seconds, whereas the Oracle takes around 50 seconds to reach a similar level. Around the 60-second mark, both methods attain comparable values, after which the LBM continues to exhibit a slight advantage. This faster convergence of the LBM confirms the approximate 50% speedup it provides on this high-dimensional problem.

\newpage

## White Wine

| Parameter | Value          |
| :-------- | -------------- |
| Kernel    | RBF(sigma=0.5) |
| C         | 1              |
| Epsilon   | 0.01           |

### Oracle
| Iterations | MSE     | Time (s) |
| ---------- | ------- | -------- |
| 60         | 0.06494 | 438.608  |

### SVR with Level Bundle Method (LBM)

- **tol**: `1e-2` 
- **theta**: `0.8`   
- **max_constraints**: `60` 

| Iterations | MSE      | Time (s) |
| ---------- | ------   | -------- |
| 60         | 0.064731 | 268.0662 |

\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/white_wine_gap.jpg}
\end{figure}

From the realtive gap plot we observe no significant differences between the two SVR models: after few initial step (where the Oracle has a steeper increase as noticed also in Abalone results) both exhibit excellent and comparable results, evidenced by the small MSE and identical convergence steps after eleventh iteration.

\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/white_wine_time.jpg}
\end{figure}

Consistent with the findings on the Abalone dataset, the Oracle exhibits a significantly slower processing time, particularly noticeable within the first minute where a substantial temporal gap exists.

However, beyond this initial period, thanks to the initially larger Oracle's steps size, the difference diminishes considerably, and the two methods proceed with similar steps within comparable timeframes. Despite this initial and substantial temporal lag, which contributes to an overall runtime for the Oracle exceeding that of the LBM by over 200 seconds, the final convergence in performance remains comparable between the two approaches.

\newpage

## Red Wine

| Parameter | Value          |
| :-------- | -------------- |
| Kernel    | RBF(sigma=0.5) |
| C         | 1              |
| Epsilon   | 0.01           |

### Oracle
| Iterations | MSE    | Time (s) |
| ---------- | ------ | -------- |
| 60         | 0.05532 | 29.7456 |

### SVR with Level Bundle Method (LBM)

- **tol**: `1e-2` 
- **theta**: `0.6`   
- **max_constraints**: `60` 

| Iterations | MSE    | Time (s) |
| ---------- | ------ | -------- |
| 60         | 0.055118 | 43.705 |

\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/wine_gap.jpg}
\end{figure}

Similar to the White Wine dataset, both SVR models exhibit comparable and excellent convergence, as evidenced by the minimal relative gap difference observed after the Oracle method's initial iteration.

\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/wine_time.jpg}
\end{figure}

Analyzing the time-based performance reveals a negligible difference between the two methods. Comparing the plot with the total runtime results, it's noticeable that in this case, the computational simplicity of the Oracle might be favored over the LBM approach, demonstrating an approximate 30% performance advantage for the Oracle in terms of overall computation time.



\newpage

## Airfoil

| Parameter | Value          |
| :-------- | -------------- |
| Kernel    | RBF(sigma=0.7) |
| C         | 1              |
| Epsilon   | 0.01           |

### Oracle
| Iterations | MSE    | Time (s) |
| ---------- | ------ | -------- |
| 60         | 10.2377 | 46.8911 |

### SVR with Level Bundle Method (LBM)

- **tol**: `1e-2` 
- **theta**: `0.7`   
- **max_constraints**: `60` 

| Iterations | MSE    | Time (s) |
| ---------- | ------ | -------- |
| 60         | 10.2376 | 43.7915 |


\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/airfoil_gap1.jpg}
\end{figure}

Similar to previous dataset observations, a noticeable difference in the relative gap between the two methods is apparent in the initial 10 iterations, with their performance tending to align from approximately 15 to 20 iterations. The Oracle exhibits an initial increase in the relative gap magnitude within the first few steps before rapidly converging towards the LBM's result, eventually reaching a plateau indicative of optimal convergence.

\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{./assets/airfoil_time1.jpg}
\end{figure}

In terms of computational time, the Oracle, thanks to its noticeably larger initial steps in function evaluation, reaches a similar function value as the LBM after roughly 10 seconds, resulting in virtually indistinguishable outputs thereafter. However, this initial temporal lag contributes to an overall performance difference, with the Oracle taking approximately 6% (3 seconds) longer to converge fully compared to the LBM.

\newpage

## Summary

| Dataset     | Model     | Iterations | MSE      | Time (s) |
| ----------- | --------- | ---------- | -------- | -------- |
| Abalone     | Oracle    | 60         | 4.1976   | 334.2511 |
| Abalone     | LBM (SVR) | 60         | 4.1966   | 148.3282 |
| White Whine | Oracle    | 60         | 0.064935 | 438.608  |
| White Whine | LBM (SVR) | 60         | 0.064731 | 268.0662 |
| Red Whine   | Oracle    | 60         | 0.05532  | 29.7456  |
| Red Whine   | LBM (SVR) | 60         | 0.055118 | 43.705   |
| Airfoil     | Oracle    | 60         | 10.2345  | 46.8911  |
| Airfoil     | LBM (SVR) | 60         | 10.2336  | 43.7915  |

Across all datasets, the Level Bundle Method (LBM) with Support Vector Regression (SVR) generally achieves a marginally lower MSE compared to the Oracle approach. However, the computational time efficiency of the two methods varies depending on the dataset size and complexity.

For large, high-dimensional datasets such as Abalone and White Wine, the Oracle requires substantially more time to converge, with differences reaching up to 50%, likely due to its more exhaustive evaluation of candidate solutions. In these scenarios, LBM (SVR) significantly reduces the runtime by approximately 45–50% while still delivering a slightly better MSE.

In contrast, for smaller or lower-dimensional datasets like Airfoil and Red Wine, the time difference between the two methods becomes less pronounced. As observed in the Airfoil dataset, the time gap is limited to around 3 seconds (6%) . Interestingly, in the Red Wine dataset (which has a higher number of features but a significantly lower number of input samples compared to Abalone and White Wine), the Oracle is actually faster than LBM (SVR) to reach 60 iterations, despite its MSE remaining marginally higher. This suggests that in cases with limited input data but higher feature dimensionality, the overhead associated with the LBM's bundle maintenance and update processes can outweigh its benefits.

Overall, LBM (SVR) appears to offer the best trade-off for prediction tasks involving a large number of input samples. However, when dealing with datasets characterized by a high number of features but a limited number of input data points, the Oracle tends to outperform LBM in terms of computational time, although its MSE is typically slightly higher. 
