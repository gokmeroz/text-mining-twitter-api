import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_tweet(text):
    # Remove URLs, mentions, hashtags, special chars
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs, mentions, hashtags, special chars
    text = re.sub(r"@\w+|#\w+", "", text)  # Remove user tags 
    text = re.sub(r"[^a-zA-Z\s]", "", text) # Remove special chars and extra, double spaces 
    text = text.lower() 
    
    # Tokenize
    tokens = word_tokenize(text)  

    # Remove stopwords and short tokens like 'a','I' 
    filtered = [word for word in tokens if word not in stop_words and len(word) > 1]

    return " ".join(filtered) # De-tokenize words 

# === APPLY TO DATASET ===
input_file = 'goat_debate.json'
output_file = 'goat_debate_cleaned.json'

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    data = json.load(fin)
    for tweet_data in data:
        try:
            cleaned_text = preprocess_tweet(tweet_data['tweet'])
            new_data = {
                'username': tweet_data['username'],
                'original_tweet': tweet_data['tweet'],
                'cleaned_tweet': cleaned_text
            }
            json.dump(new_data, fout, ensure_ascii=False)
            fout.write('\n')
        except Exception as e:
            print(f"⚠️ Skipping malformed tweet: {tweet_data} — Reason: {e}")
