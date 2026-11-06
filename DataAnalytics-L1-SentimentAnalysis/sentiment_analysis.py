import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

# Load the Twitter sentiment dataset
df = pd.read_csv("Tweets.csv/Tweets.csv")

# Display the first 5 rows
print(df.head())

# Display dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# Count each sentiment category
print("\nSentiment Distribution:")
print(df["airline_sentiment"].value_counts())

# Check for missing values in important columns
print("\nMissing values:")
print(df[["text", "airline_sentiment"]].isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())


import matplotlib.pyplot as plt
import seaborn as sns

# Count sentiments
sentiment_counts = df["airline_sentiment"].value_counts()

# Create bar chart
plt.figure(figsize=(8, 5))
sns.barplot(
    x=sentiment_counts.index,
    y=sentiment_counts.values
)

plt.title("Sentiment Distribution of Airline Tweets")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.tight_layout()

# Save chart
plt.savefig("sentiment_distribution.png", dpi=300)
plt.show()


# Download English stopwords
nltk.download("stopwords")

# Load stopwords
stop_words = set(stopwords.words("english"))

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Create a cleaned text column
df["cleaned_text"] = df["text"].apply(clean_text)

# Display original and cleaned tweets
print("\nOriginal and Cleaned Tweets:")
print(df[["text", "cleaned_text"]].head())


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Convert cleaned text into numerical features
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["cleaned_text"])

# Sentiment labels
y = df["airline_sentiment"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTF-IDF and Data Split Results:")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("TF-IDF features:", X.shape[1])


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predict sentiment on test data
y_pred_nb = nb_model.predict(X_test)

# Evaluate the model
print("\nNaive Bayes Model Results")
print("Accuracy:", round(accuracy_score(y_test, y_pred_nb), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Predict sentiment on test data
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
print("\nLogistic Regression Model Results")
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import pandas as pd

# Compare model accuracy
results = pd.DataFrame({
    "Model": ["Naive Bayes", "Logistic Regression"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_lr)
    ]
})

print("\nModel Comparison:")
print(results)

# Select the model with higher accuracy
if accuracy_score(y_test, y_pred_nb) >= accuracy_score(y_test, y_pred_lr):
    best_model = nb_model
    best_predictions = y_pred_nb
    best_name = "Naive Bayes"
else:
    best_model = lr_model
    best_predictions = y_pred_lr
    best_name = "Logistic Regression"

print("\nBest model by accuracy:", best_name)

# Create confusion matrix
labels = ["negative", "neutral", "positive"]
cm = confusion_matrix(y_test, best_predictions, labels=labels)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate a separate WordCloud for each sentiment
for sentiment in ["positive", "negative", "neutral"]:
    text_data = " ".join(
        df.loc[df["airline_sentiment"] == sentiment, "cleaned_text"]
        .dropna()
    )

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate(text_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{sentiment.title()} Sentiment WordCloud")
    plt.tight_layout()

    plt.savefig(f"wordcloud_{sentiment}.png", dpi=300)
    plt.show()
    plt.close()

print("All three WordClouds created successfully!")



# Find incorrectly classified tweets
misclassified = df.loc[y_test.index].copy()

misclassified["actual_sentiment"] = y_test
misclassified["predicted_sentiment"] = best_predictions

misclassified = misclassified[
    misclassified["actual_sentiment"]
    != misclassified["predicted_sentiment"]
]

print("\nMisclassified Tweets (First 5):")
print(
    misclassified[
        ["text", "actual_sentiment", "predicted_sentiment"]
    ].head(5).to_string(index=False)
)

# Save the examples for analysis
misclassified[
    ["text", "actual_sentiment", "predicted_sentiment"]
].head(5).to_csv("misclassified_examples.csv", index=False)

print("\nMisclassified examples saved successfully!")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Calculate evaluation metrics for both models
summary = pd.DataFrame({
    "Model": ["Naive Bayes", "Logistic Regression"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_lr)
    ],
    "Precision (weighted)": [
        precision_score(y_test, y_pred_nb, average="weighted", zero_division=0),
        precision_score(y_test, y_pred_lr, average="weighted", zero_division=0)
    ],
    "Recall (weighted)": [
        recall_score(y_test, y_pred_nb, average="weighted", zero_division=0),
        recall_score(y_test, y_pred_lr, average="weighted", zero_division=0)
    ],
    "F1-score (weighted)": [
        f1_score(y_test, y_pred_nb, average="weighted", zero_division=0),
        f1_score(y_test, y_pred_lr, average="weighted", zero_division=0)
    ]
})

print("\nFinal Model Evaluation:")
print(summary.round(4).to_string(index=False))

summary.to_csv("model_evaluation.csv", index=False)
print("\nModel evaluation saved to model_evaluation.csv")


