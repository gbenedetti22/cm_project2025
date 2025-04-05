\newpage
# Conclusions

## Hyperparameters Used

During the hyperparameter selection phase for the various SVR models, we maintained the same values as much as possible to ensure a fair and consistent comparison among the different approaches.

### SVR Parameters:
- **`kernel_function`**: `RBFKernel()` 

  *Kernel function used in the Support Vector Regression (SVR) model (Radial Basis Function Kernel in this case).*

- **`C`**: 1
  
   *Regularization parameter for the SVR model, controlling the trade-off between model complexity and training error.*
  
- **`epsilon`**: 0.05
  
  *Epsilon margin for the SVR model, defining the tolerance for prediction errors.*
  
- **`max_iter`**: 500 
  
  *Maximum number of iterations for the Level Bundle Method (LBM).*
  
- **`opt`**: `lbm` 
  
  *Optimizer used for the SVR model (Level Bundle Method in this case).*

### LBM Parameters:
- **`tol`**: 1e-3 
  
  *Tolerance for the subgradient step in the LBM (stopping criteria).*
  
- **`theta`**: 0.1
  
  *Convex combination parameter used in the LBM.*
  
- **`max_constraints`**: 50
  
  *Maximum number of constraints allowed in the bundle.*

## Results

Thanks to the optimizations introduced so far, our LBM method achieved **sustainable performance**, with a **Mean Squared Error (MSE) of 4.3729**, and a **linear convergence rate**. This result is totally comparable with the Oracle that obtained an **MSE of 4.2187**  as shown in the previous chapter.

```{=latex}
\begin{center}
\makebox[\textwidth][c]{
\includegraphics[width=1.3\textwidth]{./assets/lbm_abalone_results.jpg}
}
\end{center}
```

From the analysis of the convergence plots of the SVR with the **Level Bundle method**, several significant conclusions can be drawn: 

- The **objective function** plot shows a **steady and monotonic decrease** until approximately *-3000*, indicating that the optimization is effectively progressing toward minimization.

- The **gradient norm** stabilizes around *745* after about *150* iterations, suggesting the attainment of a **stationary point**. 
- The **distribution of step norms** reveals that the algorithm prefers step sizes between *0.05* and *0.15*, with a higher concentration around *0.075*, indicating a balanced behavior between exploration and exploitation of the solution space. 
- Particularly interesting is the relationship between **gradient norm** and **step norm**, which shows two significant peaks around iteration *50*, suggesting that during that phase, the algorithm traversed regions with **strong curvature** or **gradient discontinuities**. 

Overall, the **Level Bundle method** demonstrates good computational efficiency, achieving convergence in approximately *150* iterations with a stable trend in the objective function, characteristics that make it well-suited for complex optimization problems such as those addressed in the SVR context.
