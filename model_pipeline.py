import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# === STEP 1: Load and Label Cleaned Tweets ===
input_file = 'goat_debate_cleaned.json'
tweets, labels = [], []

keywords_lebron = ['lebron', 'king james', 'bron']
keywords_jordan = ['jordan', 'mj', 'michael jordan']

def label_tweet(text):
    text = text.lower()
    if any(k in text for k in keywords_lebron) and not any(k in text for k in keywords_jordan):
        return 'lebron'
    elif any(k in text for k in keywords_jordan) and not any(k in text for k in keywords_lebron):
        return 'jordan'
    else:
        return 'neutral'

with open(input_file, 'r', encoding='utf-8') as fin:
    for line in fin:
        data = json.loads(line)
        tweet = data['cleaned_tweet']
        label = label_tweet(tweet)
        tweets.append(tweet)
        labels.append(label)

print("Label distribution:", Counter(labels))

# === STEP 2: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    tweets, labels, test_size=0.2, random_state=42, stratify=labels
)

# === STEP 3: TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === STEP 4: Train Models ===

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_preds = lr_model.predict(X_test_vec)

# === STEP 5: Evaluation ===
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_true, y_pred, digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=['lebron', 'jordan', 'neutral'])
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['lebron', 'jordan', 'neutral'],
                yticklabels=['lebron', 'jordan', 'neutral'],
                cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

evaluate_model("Naive Bayes", y_test, nb_preds)
evaluate_model("Logistic Regression", y_test, lr_preds)