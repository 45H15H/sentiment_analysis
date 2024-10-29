# app.py
from flask import Flask, request, jsonify
from textblob import TextBlob

app = Flask(__name__)

@app.route('/', methods=['POST'])
def analyze_sentiment():
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "score": sentiment_score
    })

if __name__ == "__main__":
    app.run()
