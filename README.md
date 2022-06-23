# package delivery optimisation problem
This is a project for the course of "Mathematical Optimisation" at University of Trieste.

The aim of the project is to reproduce the mathematical optimization model presented in this original [paper](paper.pdf)

## Structure
All the .py files are class file or executor that help to condensate the code of the notebooks.

The original [paper](paper.pdf) presents two mathematical optimization model:
  - A single period version:
  
    that is implemented in [single_period_problem_gurobi](single_period_problem_gurobi.ipynb) and [single_period_problem_xpress](single_period_problem_xpress.ipynb) in which two different solver are used
  - A multi period version:
  
    that is implemented in [multi_period_problem_gurobi](multi_period_problem_gurobi.ipynb)
    
  Then there are two notebook [scability_single_period_analysis](scability_single_period_analysis.ipynb) and [scalability_multi_period_analysis](scalability_multi_period_analysis.ipynb) that illustrate how the model behave when the instance of the problem increase
