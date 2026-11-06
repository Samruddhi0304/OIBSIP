
import pandas as pd

# Load the advertising dataset
df = pd.read_csv("advertising.csv")

# Display the first 5 rows
print("First 5 Rows:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize relationships between advertising channels and sales
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.scatterplot(data=df, x="TV", y="Sales", ax=axes[0])
axes[0].set_title("TV Advertising vs Sales")

sns.scatterplot(data=df, x="Radio", y="Sales", ax=axes[1])
axes[1].set_title("Radio Advertising vs Sales")

sns.scatterplot(data=df, x="Newspaper", y="Sales", ax=axes[2])
axes[2].set_title("Newspaper Advertising vs Sales")

plt.tight_layout()
plt.savefig("advertising_vs_sales.png", dpi=300)
#plt.show()

# Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(
    df[["TV", "Radio", "Newspaper", "Sales"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Advertising and Sales Correlation")
plt.tight_layout()
plt.savefig("sales_correlation_heatmap.png", dpi=300)
#plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Define input features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Split data into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict sales
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2 = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Results")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R-squared:", round(r2, 3))

print("\nAdvertising Coefficients:")
for feature, coefficient in zip(X.columns, lr_model.coef_):
    print(f"{feature}: {coefficient:.3f}")

print("Intercept:", round(lr_model.intercept_, 3))


from sklearn.ensemble import RandomForestRegressor

# Train Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Results")
print("MAE:", round(mae_rf, 3))
print("RMSE:", round(rmse_rf, 3))
print("R-squared:", round(r2_rf, 3))

# Feature importance
print("\nFeature Importance:")
for feature, importance in zip(X.columns, rf_model.feature_importances_):
    print(f"{feature}: {importance:.3f}")


# Compare Linear Regression and Random Forest

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [mae, mae_rf],
    "RMSE": [rmse, rmse_rf],
    "R2 Score": [r2, r2_rf]
})

print("\nModel Comparison")
print(comparison.round(3))

# Save comparison to CSV
comparison.to_csv("model_comparison.csv", index=False)
print("\nModel comparison saved successfully!")


# Residual Plot for Linear Regression

residuals = y_test - y_pred_lr

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred_lr, y=residuals)

plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot - Linear Regression")

plt.tight_layout()
plt.savefig("residual_plot.png", dpi=300)
#plt.show()

print("\nResidual plot saved successfully!")


# Identify the most influential advertising channel

importance_df = pd.DataFrame({
    "Advertising Channel": X.columns,
    "Linear Regression Coefficient": lr_model.coef_,
    "Random Forest Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Random Forest Importance",
    ascending=False
)

print("\nAdvertising Channel Impact")
print(importance_df.round(3))

# Save results
importance_df.to_csv("advertising_channel_impact.csv", index=False)

print("\nAdvertising channel impact saved successfully!")
