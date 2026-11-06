import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create folder for graphs
os.makedirs("screenshots", exist_ok=True)

# Load dataset
df = pd.read_csv("train.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\nStatistical Summary:")
print(df.describe())

# Target variable distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["SalePrice"], kde=True)
plt.title("House Sale Price Distribution")
plt.savefig("screenshots/price_distribution.png")
plt.close()

# Correlation heatmap
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(12, 9))
sns.heatmap(numeric_df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("screenshots/correlation_heatmap.png")
plt.close()

# Separate features and target
X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
y = df["SalePrice"]

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(exclude=np.number).columns

# Handle missing values and encode categories
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Create model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# Actual vs Predicted plot
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.savefig("screenshots/actual_vs_predicted.png")
plt.close()

# Residual plot
residuals = y_test - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("screenshots/residual_plot.png")
plt.close()

# Feature coefficient analysis
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
coefficients = model.named_steps["regressor"].coef_

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

coef_df["Absolute Impact"] = coef_df["Coefficient"].abs()
top_features = coef_df.nlargest(20, "Absolute Impact")

plt.figure(figsize=(10, 8))
sns.barplot(data=top_features, x="Coefficient", y="Feature")
plt.title("Top 20 Feature Impacts")
plt.tight_layout()
plt.savefig("screenshots/feature_impact.png")
plt.close()

# Save predictions
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

results.to_csv("house_price_predictions.csv", index=False)

print("\nProject completed successfully!")
print("Graphs saved in screenshots folder.")
print("Predictions saved in house_price_predictions.csv")