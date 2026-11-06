# Sales Prediction Using Python

## Project Overview
This project analyzes the relationship between advertising expenditure and product sales. Machine learning models are used to predict sales based on advertising budgets.

## Dataset
The project uses the Advertising dataset containing:
- TV advertising expenditure
- Radio advertising expenditure
- Newspaper advertising expenditure
- Sales

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow
1. Load and explore the dataset.
2. Check missing values and descriptive statistics.
3. Visualize advertising expenditure against sales.
4. Analyze correlations between advertising channels and sales.
5. Train a Linear Regression model.
6. Train a Random Forest Regressor model.
7. Evaluate both models using MAE, RMSE, and R-squared.
8. Compare model performance.
9. Visualize residuals.
10. Analyze advertising channel importance.

## Output Files
- `advertising_vs_sales.png` — Advertising and sales scatter plots
- `sales_correlation_heatmap.png` — Correlation heatmap
- `model_comparison.csv` — Comparison of model performance
- `residual_plot.png` — Linear Regression residual plot
- `advertising_channel_impact.csv` — Advertising channel impact analysis

## How to Run
Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the script:

```bash
python sales_prediction.py
```

## Conclusion
The project compares two regression models and examines the relationship between advertising expenditure and sales. The evaluation metrics help determine which model performs better on the test dataset.
