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
            f.write(f"RMSE: {np.sqrt(mse):.4f}\n")
            f.write(f"R Squared: {r2_score(y, y_pred):.4f}\n")
        
            # predictions
            df = pd.DataFrame(y_pred, columns=['Predicted Y'])
            df.to_csv(f"Results/{file[:-4]}_pred.csv", index=False)

            if file != '3D H.csv':
                f.write("\n")

        except FileNotFoundError:
            print(f"Error: file not found in directory. Try again.")