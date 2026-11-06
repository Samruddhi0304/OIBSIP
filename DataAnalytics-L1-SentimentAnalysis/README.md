# Sentiment Analysis of Airline Tweets

## Project Overview
This project uses Natural Language Processing (NLP) and Machine Learning to classify airline tweets into three sentiment categories: positive, negative, and neutral.

## Dataset
- **Dataset:** Twitter US Airline Sentiment
- **Total records:** 14,640 tweets
- **Features:** 15 columns
- **Target column:** `airline_sentiment`
- **Sentiment classes:** Positive, Negative, Neutral

## Technologies Used
- Python
- Pandas
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- WordCloud

## Project Workflow
1. Loaded and inspected the dataset.
2. Examined sentiment distribution, missing values, and duplicate rows.
3. Created a sentiment distribution bar chart.
4. Cleaned tweet text using lowercase conversion, URL and mention removal, punctuation removal, and stopword removal.
5. Converted text into numerical features using TF-IDF.
6. Split the dataset into training and testing sets using an 80:20 ratio.
7. Trained Naive Bayes and Logistic Regression classifiers.
8. Evaluated both models using accuracy, precision, recall, and F1-score.
9. Generated a confusion matrix and WordClouds for each sentiment.
10. Examined five misclassified tweet examples.

## Output Files
- `sentiment_distribution.png`
- `confusion_matrix.png`
- `wordcloud_positive.png`
- `wordcloud_negative.png`
- `wordcloud_neutral.png`
- `misclassified_examples.csv`
- `model_evaluation.csv`

## Results
The models are evaluated using accuracy, precision, recall, and F1-score. Refer to `model_evaluation.csv` for the actual evaluation results.

## Conclusion
This project demonstrates how Natural Language Processing and Machine Learning can be used to analyze sentiment in airline tweets. The comparison of Naive Bayes and Logistic Regression helps identify which model performs better on the test dataset.

## How to Run
1. Install the required libraries.
2. Place the dataset in the expected project folder.
3. Run `python sentiment_analysis.py` from the project directory.

## Level 2 - Task 1: House Price Prediction

[View House Price Prediction Project](DataAnalytics-L2-HousePricePrediction/README.md)
