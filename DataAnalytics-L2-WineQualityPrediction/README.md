# Wine Quality Prediction Using Machine Learning

## Project Overview
This project predicts red wine quality using machine learning classification algorithms. The quality of wine is classified into two categories: Good and Bad.

## Dataset
- **Dataset:** Wine Quality Red Dataset
- **Total Records:** 1599
- **Total Columns:** 12
- **Target Variable:** Quality
- **Dataset Source:** [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow
1. Load and explore the dataset.
2. Perform Exploratory Data Analysis (EDA).
3. Check missing values and class distribution.
4. Create a correlation heatmap.
5. Convert wine quality into Good and Bad classes.
6. Perform stratified train-test split.
7. Apply feature scaling.
8. Train Random Forest, SGD and SVC models.
9. Evaluate models using accuracy, precision, recall and F1-score.
10. Generate confusion matrices.
11. Compare model performance.
12. Analyze Random Forest feature importance.

## Model Performance

| Model | Accuracy |
|---|---:|
| Random Forest | 93.13% |
| SVC | 84.69% |
| SGD Classifier | 80.31% |

## Feature Importance
Alcohol was the most important feature, followed by sulphates and volatile acidity.

## Visualizations
The project includes:
- Feature distribution plots
- Correlation heatmap
- Confusion matrices
- Model accuracy comparison
- Feature importance chart

## Conclusion
Random Forest achieved the highest accuracy of 93.13% among the three tested models. This project demonstrates the application of machine learning classification techniques for predicting wine quality.