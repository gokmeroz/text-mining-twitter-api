import json
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from wordcloud import WordCloud
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import nltk

# Download tokenizer
nltk.download('punkt')

# === Load Cleaned Data ===
input_file = 'goat_debate_cleaned.json'
tweets = []

with open(input_file, 'r', encoding='utf-8') as fin:
    for line in fin:
        data = json.loads(line)
        tweets.append(data['cleaned_tweet'])

# === Word Frequency ===
all_words = ' '.join(tweets).split()
word_counts = Counter(all_words)
most_common = word_counts.most_common(20)

# Plot Top Words
words, counts = zip(*most_common)
plt.figure(figsize=(10, 6))
sns.barplot(x=list(counts), y=list(words))
plt.title("Top 20 Most Common Words in Tweets")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.tight_layout()
plt.show()

# === Word Cloud ===
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tweets))
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Tweets")
plt.show()

# === Word2Vec Vectorization ===

# Tokenize tweets
tokenized_tweets = [word_tokenize(tweet.lower()) for tweet in tweets]

# Train Word2Vec model
w2v_model = Word2Vec(sentences=tokenized_tweets, vector_size=100, window=5, min_count=2, workers=4)

# Convert each tweet to vector by averaging word embeddings
def tweet_to_vector(tokens, model):
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if valid_vectors:
        return np.mean(valid_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

tweet_vectors = np.array([tweet_to_vector(tokens, w2v_model) for tokens in tokenized_tweets])

# === Output vector shape and sample ===
print("Tweet vectors shape:", tweet_vectors.shape)
print("Sample tweet vector (first tweet):", tweet_vectors[0])
