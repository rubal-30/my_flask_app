from flask import Flask, request, jsonify
import pandas as pd
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import requests
import io

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    return text

# Function to fetch CSV from GitHub
def fetch_csv_from_github(repo_url):
    response = requests.get(repo_url)
    response.raise_for_status()  # Ensure we notice bad responses
    data = response.content.decode('utf-8')
    return pd.read_csv(io.StringIO(data))

@app.route('/analyze_sentiment/<fort_name>', methods=['GET'])
def analyze_sentiment(fort_name):
    repo_url = f'https://raw.githubusercontent.com/rubal-30/my_flask_app/main/{fort_name}.csv'  # Update if necessary

    try:
        df = fetch_csv_from_github(repo_url)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    df['clean_review'] = df['wiI7pd'].apply(preprocess_text)
    df['sentiment'] = df['clean_review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    mean_sentiment = df['sentiment'].mean()

    if (mean_sentiment is None):
        overall_sentiment = 'unknown'
    elif mean_sentiment >= 0.5:
        overall_sentiment = 'easy'
    elif mean_sentiment >= 0:
        overall_sentiment = 'moderate'
    else:
        overall_sentiment = 'tough'

    return jsonify({"fort_name": fort_name, "overall_sentiment": overall_sentiment})

if __name__ == '__main__':
    app.run(debug=True)
