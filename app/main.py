from flask import Flask, request, jsonify
import re
import string
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import json

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/trained_pickle.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open(f"{BASE_DIR}/vectorizer.pkl", 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Загрузка данных о сайтах из JSON-файла
with open(f"{BASE_DIR}/websites.json", 'r') as json_file:
    websites_data = json.load(json_file)

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    
    return text

def preprocess_text(text):
    cleaned_text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s.,?!]', '', text)
    return cleaned_text

def is_website_matching(website, website_list):
    # Проверяем, содержится ли часть домена в списке сайтов
    for stored_website in website_list:
        if stored_website.lower().strip() in website.lower().strip():
            return True
    return False

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        app.logger.info(f"Received data: {data}")

        if 'text' in data:
            news = data['text']
        elif 'texts' in data:
            news = " ".join(data['texts'])
        else:
            raise ValueError("Invalid input format. 'text' or 'texts' field is required.")

        app.logger.info(f"Original text: {news}")

        news = preprocess_text(news)
        news = wordopt(news)
        news_vectorized = vectorizer.transform([news])
        prediction = model.predict(news_vectorized)

        # Проверяем, есть ли поле 'website' в данных
        if 'website' in data:
            website_name = data['website']
            # Проверяем, содержится ли часть домена в списке сайтов из JSON-файла
            is_website_matching_result = is_website_matching(website_name, websites_data["websites"])
        else:
            is_website_matching_result = None

        result = {
            "prediction": int(prediction[0]),
            "website_matching": is_website_matching_result
        }
        app.logger.info(f"Prediction: {result}")

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
