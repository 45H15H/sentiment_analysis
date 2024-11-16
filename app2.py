# app.py
from flask import Flask, request, jsonify
from textblob import TextBlob
import email
from email import policy
from email.parser import BytesParser

app2 = Flask(__name__)

@app2.route('/', methods=['POST'])
def analyze_sentiment():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file.filename.endswith('.eml'):
        return jsonify({"error": "Invalid file type"}), 400

    msg = BytesParser(policy=policy.default).parse(file.stream)
    if msg.is_multipart():
        for part in msg.iter_parts():
            if part.get_content_type() == 'text/plain':
                text = part.get_payload(decode=True).decode(part.get_content_charset())
                break
        else:
            return jsonify({"error": "No text/plain part found in the email"}), 400
    else:
        text = msg.get_payload(decode=True).decode(msg.get_content_charset())

    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "score": sentiment_score
    })

if __name__ == "__main__":
    app2.run()