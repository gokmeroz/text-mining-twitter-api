import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

# === STEP 1: Load Cleaned Tweets and Label ===
def label_tweet(text):
    keywords_lebron = ['lebron', 'king james', 'bron']
    keywords_jordan = ['jordan', 'mj', 'michael jordan']
    text = text.lower()
    if any(k in text for k in keywords_lebron) and not any(k in text for k in keywords_jordan):
        return 'lebron'
    elif any(k in text for k in keywords_jordan) and not any(k in text for k in keywords_lebron):
        return 'jordan'
    else:
        return 'neutral'

tweets, labels = [], []
with open('goat_debate_cleaned.json', 'r', encoding='utf-8') as f:
    for line in f:
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

# === STEP 3: Vectorization ===
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === STEP 4: Train Models and Predict ===
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_preds = lr_model.predict(X_test_vec)

# === STEP 5: Unified Evaluation Function ===
def evaluate_model(y_true, y_pred, model_name):
    labels_order = ['lebron', 'jordan', 'neutral']
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_order, yticklabels=labels_order)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.show()

    print(f"\n{model_name} - Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=labels_order, digits=3))

# === STEP 6: Evaluate All Models ===
evaluate_model(y_test, nb_preds, "Naive Bayes")
evaluate_model(y_test, lr_preds, "Logistic Regression")
