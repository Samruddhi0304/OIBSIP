
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

os.makedirs("screenshots", exist_ok=True)

# 1. Load Dataset
df = pd.read_csv("winequality-red.csv", sep=";")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nQuality Distribution:")
print(df["quality"].value_counts().sort_index())

# 2. Exploratory Data Analysis
df.hist(figsize=(14, 10), bins=20)
plt.suptitle("Wine Chemical Properties Distribution")
plt.tight_layout()
plt.savefig("screenshots/feature_distributions.png")
plt.close()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Wine Quality Correlation Heatmap")
plt.tight_layout()
plt.savefig("screenshots/correlation_heatmap.png")
plt.close()

# 3. Convert Quality into Good and Bad Classes
# Quality >= 7 = Good, Quality < 7 = Bad
df["quality_label"] = (df["quality"] >= 7).astype(int)

print("\nClass Distribution:")
print(df["quality_label"].value_counts())
print("0 = Bad Wine, 1 = Good Wine")

# 4. Separate Features and Target
X = df.drop(columns=["quality", "quality_label"])
y = df["quality_label"]

# 5. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train Three Models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),
    "SGD Classifier": SGDClassifier(
        random_state=42,
        class_weight="balanced",
        max_iter=2000
    ),
    "SVC": SVC(
        class_weight="balanced"
    )
}

results = []
trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

    trained_models[name] = model

    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Bad Wine", "Good Wine"],
        zero_division=0
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Bad", "Good"],
        yticklabels=["Bad", "Good"]
    )
    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(
        "screenshots/" + name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )
    plt.close()

# 8. Compare Model Performance
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\nMODEL COMPARISON:")
print(results_df)

results_df.to_csv("model_comparison.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("screenshots/model_comparison.png")
plt.close()

# 9. Random Forest Feature Importance
rf_model = trained_models["Random Forest"]

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)

plt.figure(figsize=(9, 6))
sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("screenshots/feature_importance.png")
plt.close()

importance_df.to_csv("feature_importance.csv", index=False)

print("\nProject Completed Successfully!")
print("All graphs saved in screenshots folder.")
