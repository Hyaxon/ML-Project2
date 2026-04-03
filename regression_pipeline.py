# Load the dataset
# Fit regression
# Save coefficient, intercept, prediction, and metrics values
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

datasets = [
    '1D L.csv', '1D M.csv', '1D H.csv',
    '2D L.csv', '2D M.csv', '2D H.csv',
    '3D L.csv', '3D M.csv', '3D H.csv'
]

os.makedirs('Results', exist_ok=True)
summary_rows = []

with open('Results/results.txt', 'w') as f:

    for file in datasets:
        try:
            # load data and split
            df = pd.read_csv(os.path.join('Data', file), header=None)

            X = df.iloc[:, :-1].values  # all rows and all cols but last
            y = df.iloc[:, -1].values   # all rows and last col

            model = LinearRegression()
            model.fit(X,y)
            y_pred = model.predict(X)

            f.write(f"Results for {file} dataset:\n")

            # intercept & coefficient(s)
            f.write(f"Intercept: {model.intercept_:.4f}\n")
            for i, coef in enumerate(model.coef_):
                f.write(f"Beta_{i+1}: {coef:.4f}\n")
            
            # metrics
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)

            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"R Squared: {r2:.4f}\n")
        
            # predictions
            df = pd.DataFrame(y_pred, columns=['Predicted Y'])
            df.to_csv(f"Results/{file[:-4]}_pred.csv", index=False)

            row = {
                'dataset': file[:-4],
                'num_predictors': X.shape[1],
                'intercept': model.intercept_,
                'beta_1': model.coef_[0] if len(model.coef_) > 0 else np.nan,
                'beta_2': model.coef_[1] if len(model.coef_) > 1 else np.nan,
                'beta_3': model.coef_[2] if len(model.coef_) > 2 else np.nan,
                'rmse': rmse,
                'r2': r2
            }
            summary_rows.append(row)

            if file != '3D H.csv':
                f.write("\n")

        except FileNotFoundError:
            print(f"Error: file not found in directory. Try again.")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('Results/model_summary.csv', index=False)    