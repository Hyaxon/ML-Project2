# Use the outputs of the regression to make plots and summary tables

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = 'Data'
results_dir = 'Results'
figures_dir = 'Figures'

os.makedirs(figures_dir, exist_ok=True)

datasets = [
    '1D L.csv', '1D M.csv', '1D H.csv',
    '2D L.csv', '2D M.csv', '2D H.csv',
    '3D L.csv', '3D M.csv', '3D H.csv'
]

summary_path = os.path.join(results_dir, 'model_summary.csv')
summary_df = pd.read_csv(summary_path)

# Read noise levels 
summary_df['noise_level'] = summary_df['dataset'].str.split().str[1]
summary_df['noise_level'] = pd.Categorical(
    summary_df['noise_level'],
    categories=['L', 'M', 'H'],
    ordered=True
)
summary_df = summary_df.sort_values(by=['num_predictors', 'noise_level'])

summary_df.to_csv(os.path.join(results_dir, 'model_summary.csv'), index=False)

for file in datasets:
    dataset_name = file[:-4]

    data_path = os.path.join(data_dir, file)
    pred_path = os.path.join(results_dir, f'{dataset_name}_pred.csv')

    try:
        # Load original data
        df = pd.read_csv(data_path, header=None)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Load predictions
        pred_df = pd.read_csv(pred_path)
        y_pred = pred_df['Predicted Y']

        # Residuals
        residuals = y - y_pred
        num_predictors = X.shape[1]

        # 1D dataset scatter and regression line
        if num_predictors == 1:
            x_vals = X.iloc[:, 0].values
            sort_idx = np.argsort(x_vals)

            plt.figure(figsize=(8, 5))
            plt.scatter(x_vals, y, label='Actual Data')
            plt.plot(x_vals[sort_idx], y_pred.iloc[sort_idx], label='Regression Line')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'{dataset_name} - Data and Regression Line')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f'{dataset_name}_fit.png'))
            plt.close()

        # 2D dataset 3D scatter and regression plane
        elif num_predictors == 2:
            x1 = X.iloc[:, 0].values
            x2 = X.iloc[:, 1].values

            # Create grid for plane
            x1_grid, x2_grid = np.meshgrid(
                np.linspace(x1.min(), x1.max(), 25),
                np.linspace(x2.min(), x2.max(), 25)
            )

            # Get coefficients 
            row = summary_df[summary_df['dataset'] == dataset_name].iloc[0]
            intercept = row['intercept']
            beta_1 = row['beta_1']
            beta_2 = row['beta_2']

            y_grid = intercept + beta_1 * x1_grid + beta_2 * x2_grid

            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x1, x2, y, label='Actual Data')
            ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5)

            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('Y')
            ax.set_title(f'{dataset_name} - Regression Plane')

            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f'{dataset_name}_plane.png'))
            plt.close()

        # 3D dataset pair plot
        elif num_predictors == 3:
            pair_df = df.copy()
            pair_df.columns = ['X1', 'X2', 'X3', 'Y']

            pair_plot = sns.pairplot(pair_df)
            pair_plot.figure.suptitle(f'{dataset_name} - Pair Plot', y=1.02)
            pair_plot.savefig(os.path.join(figures_dir, f'{dataset_name}_pairplot.png'))
            plt.close()

        # Actual vs Predicted plot for every dataset
        plt.figure(figsize=(8, 5))
        plt.scatter(y, y_pred)
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
        plt.xlabel('Actual Y')
        plt.ylabel('Predicted Y')
        plt.title(f'{dataset_name} - Actual vs Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'{dataset_name}_actual_vs_pred.png'))
        plt.close()

        # Residual plot for every dataset
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals)
        plt.axhline(0, linestyle='--')
        plt.xlabel('Predicted Y')
        plt.ylabel('Residuals')
        plt.title(f'{dataset_name} - Residual Plot')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'{dataset_name}_residuals.png'))
        plt.close()

    except FileNotFoundError:
        print(f'Missing file for {dataset_name}')

# Comparison plots
# RMSE for each dataset
plt.figure(figsize=(10, 6))
plt.bar(summary_df['dataset'], summary_df['rmse'])
plt.xlabel('Dataset')
plt.ylabel('RMSE')
plt.title('RMSE by Dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'rmse_by_dataset.png'))
plt.close()

# R2 Values for each dataset
plt.figure(figsize=(10, 6))
plt.bar(summary_df['dataset'], summary_df['r2'])
plt.xlabel('Dataset')
plt.ylabel('R Squared')
plt.title('R Squared by Dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'r2_by_dataset.png'))
plt.close()

print("Analysis complete. Plots and report summary saved.")