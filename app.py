from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
model = joblib.load('models/fake_news_detection_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user's input text from the form
    article_text = request.form['article_text']

    # Transform the input text
    transformed_text = transform_text(article_text)

    # Make predictions using your model
    prediction = model.predict(transformed_text)

    # Return the prediction as a response
    return render_template('index.html', prediction=prediction[0])

# Function to preprocess text
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove special characters and digits using regular expressions
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Join tokens back into a single string
    processed_text = ' '.join(tokens)
    
    return processed_text

tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')  # Replace with the actual filename

# Function to transform input text using the TF-IDF vectorizer
def transform_text(input_text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(input_text)
    
    # Transform the preprocessed text using the TF-IDF vectorizer
    transformed_text = tfidf_vectorizer.transform([preprocessed_text])
    
    return transformed_text


if __name__ == '__main__':
    app.run(debug=True)
