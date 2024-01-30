from flask import Flask, render_template, request
import joblib
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = joblib.load('models/fake_news_detection_model.pkl')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    article_text = request.form['article_text']

    # Preprocess and transform the text
    transformed_text = transform_text(article_text)

    # Make prediction
    prediction = model.predict(transformed_text)[
        0]  # Extract single prediction

    return render_template('index.html', prediction=prediction)


def preprocess_text(text):
    text = text.lower()
    processed_text = ""
    for char in text:
        if char.isalpha() or char.isspace():
            processed_text += char
    return processed_text


def transform_text(input_text):
    preprocessed_text = preprocess_text(input_text)
    transformed_text = tfidf_vectorizer.transform([preprocessed_text])
    return transformed_text


if __name__ == '__main__':
    app.run(debug=True)
