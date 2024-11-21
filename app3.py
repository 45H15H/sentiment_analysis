# app.py
from flask import Flask, request, jsonify
from textblob import TextBlob
import email
from email import policy
from email.parser import BytesParser
import google.generativeai as genai
import json
import os

app3 = Flask(__name__)

# Configure Gemini API with direct API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def extract_details_with_gemini(text):
    prompt = f"""
    Extract these details from the email text and return as JSON with JSON syntax:
    - customer_name
    - order_id (if available)
    - feedback_category (must be: compliments/suggestions/queries/complaints)
    
    Email: {text}
    """
    
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except:
        return {
            "customer_name": None,
            "order_id": None, 
            "feedback_category": None
        }

@app3.route('/', methods=['POST'])
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

    # Get sentiment
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

    # Get extracted details
    extracted = extract_details_with_gemini(text)

    # Combined flat JSON response
    return jsonify({
        "sentiment": sentiment,
        "score": sentiment_score,
        "customer_name": extracted.get("customer_name"),
        "order_id": extracted.get("order_id"),
        "feedback_category": extracted.get("feedback_category")
    })

if __name__ == "__main__":
    app3.run()