from flask import Flask, request, jsonify
import re
import string
import pickle
import nltk
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/trained_pickle.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open(f"{BASE_DIR}/vectorizer.pkl", 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

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

@app.route('/predict', methods=['GET',"POST"])
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
        
        result = {"prediction": int(prediction[0])}
        app.logger.info(f"Prediction: {result}")

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)