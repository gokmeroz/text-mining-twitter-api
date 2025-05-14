# 🏀 Text Mining Twitter API – Who is the GOAT? (LeBron vs. Jordan)

This project analyzes Twitter sentiment to explore one of the most debated questions in sports history:  
**Who is the greatest basketball player of all time (GOAT) — LeBron James or Michael Jordan?**

Using data mining, NLP, and machine learning, we classify tweets into two categories:
- ✅ Pro-LeBron  
- ✅ Pro-Jordan
  
## 📚 Techniques Used
Tweepy – Twitter API v2 access

NLTK – Stopword removal, lemmatization

Gensim – Word2Vec embeddings

Scikit-learn – ML models and vectorizers

Matplotlib / Seaborn – Data visualization

---

## 📌 Features

- Collects real-time tweets using **Tweepy** and the Twitter API v2
- Cleans and preprocesses tweet text for sentiment analysis
- Transforms text using **TF-IDF** and **Word2Vec**
- Trains and evaluates two ML models: **Naive Bayes** and **Logistic Regression**
- Visualizes results with confusion matrices and classification metrics

---
## 📁 File Structure

text-mining-twitter-api/
├── collectData.py               # Twitter API scraping
├── preprocess.py                # Data cleaning & tokenization
├── vectorization.py             # TF-IDF and Word2Vec vectorization
├── model_pipeline.py            # Model training and evaluation
├── confusion_classification.py  # Confusion matrix visualization
├── goat_debate.json             # Collected tweet data
└── README.md                    # Project documentation


## 🚀 How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/gokmeroz/text-mining-twitter-api.git
cd text-mining-twitter-api
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip3 install tweepy nltk gensim scikit-learn matplotlib seaborn

### 1. Clone the Repository
python3 collectData.py
python3 preprocess.py
python3 vectorization.py
python3 model_pipeline.py
python3 confusion_classification.py

# Example Output
TF-IDF + Logistic Regression:
Accuracy: 84%
Precision: 0.83
Recall: 0.85
F1 Score: 0.84

Confusion Matrix:
[[43  7]
 [ 5 45]]





