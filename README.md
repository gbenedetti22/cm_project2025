- # 🔧 SVR with Level Bundle Method and General-Purpose Solver

  ## 📌 Project Description

  This project explores a Support Vector Regression (SVR)-type model (**M**) by applying and comparing two optimization strategies:

  - **(M)**: An SVR-type approach using one or more user-defined kernels (e.g., linear, RBF, polynomial) 🧠📈
  - **(A1)**: A **level bundle method** applied to the **dual** formulation of the SVR. This method uses an off-the-shelf solver (MOSEK) to solve the Master Problem in each iteration 🧮⚙️
  - **(A2)**: A **general-purpose solver (called Oracle)**  is applied to a suitable reformulation of the SVR problem, serving as a benchmark for comparison 📊

  The aim is to assess the performance and effectiveness of the level bundle method against more traditional optimization methods when solving kernel-based SVR problems.

  ------

  ## ▶️ How to Run the Project

  1. ✅ **Install MOSEK**
      Ensure **MOSEK 11 or later** is installed and correctly set up with MATLAB.

  2. 📂 **Open the Main Script**
      Launch MATLAB and open the file:
      `test_main.m`

  3. 📑 **Select a Dataset**
      Inside the script, choose one of the available datasets from the listed options.

  4. 🚀 **Run the Script**

     Have fun! 😃

  ------

  ## 📄 Report

  For a detailed explanation of the methodology, mathematical formulations, and experimental results, see the full project report:
   👉 [Report - cm_project2025.pdf](report/cm_project2025.pdf)

  ------

  ## 💻 Requirements

  - MATLAB
  - [MOSEK](https://www.mosek.com/) version 11 or higher
  - Optimization Toolbox (optional, for general-purpose solver support)

  ------