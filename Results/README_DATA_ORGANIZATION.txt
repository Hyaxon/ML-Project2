Project 2 Data Description

The uploaded CSVs contain the output data generated from the linear regression analysis.

Files:

- model_summary.csv:
  Contains a summary of all regression models. Each row corresponds to one dataset and includes:
  dataset name, number of predictors, intercept, coefficients (beta values), RMSE, and R^2.

- *_pred.csv files:
  Each file contains the predicted values (y_hat) for the corresponding dataset.
  For example:
    - 1D L_pred.csv corresponds to predictions for the 1D L dataset
    - 2D M_pred.csv corresponds to predictions for the 2D M dataset

These files were generated using the regression pipeline and were used to create the plots and analysis presented in the report.