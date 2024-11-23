# app.py
from flask import Flask, request, jsonify
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import email
from email import policy
from email.parser import BytesParser
import google.generativeai as genai
import json
from datetime import datetime
import re

app3 = Flask(__name__)

# Configure Gemini API with direct API key
GOOGLE_API_KEY = "AIzaSyAh2OjBYiM5rsIc-rGiTUSQuBl-xAeYWpA"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def extract_details_with_gemini(text):
    prompt = f"""
    Extract these details from the email text and return as JSON object:
    - customer_name
    - order_id (if available) else return "N/A
    - feedback_category (must be: pricing, product, suggestion, technical, packaging, quality, delivery, or service)
    - feedback_summary: write a summary of the feedback in few sentences
    - action_needed: write what action is needed by the team in response to the feedback 

    Email: {text}

    Don't use ```json``` or ```return``` in your response. Just write the JSON object as plain text.
    """
    
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except:
        return {
            "customer_name": None,
            "order_id": None, 
            "feedback_category": None,
            "feedback_summary": None,
            "action_needed": None
        }
def extract_details_from_eml(msg):
    subject = msg.get('subject')
    from_header = msg.get('from', '')
    
    # Extract email and name from From header
    email_match = re.search(r'<(.+?)>', from_header)
    customer_email = email_match.group(1) if email_match else from_header

    # Extract date and time of the email
    date_str = msg.get('date')
    try:
        # Parse the email date string
        date_obj = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
        # Format to show only the date
        formatted_date = date_obj.strftime('%Y-%m-%d')
    except:
        formatted_date = date_str  # Fallback to original date string if parsing fails

    return {
        "subject": subject,
        "customer_email": customer_email,
        "date": formatted_date
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

    # Get sentiment using both TextBlob and VADER
    # TextBlob analysis
    blob = TextBlob(text)
    tb_score = blob.sentiment.polarity
    
    # VADER analysis
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = analyzer.polarity_scores(text)
    compound_score = vader_scores['compound']
    
    # Combine scores for more accurate results
    final_score = (tb_score + compound_score) / 2
    
    # More granular sentiment classification
    if final_score >= 0.5:
        sentiment = "Very Positive"
    elif 0.1 <= final_score < 0.5:
        sentiment = "Moderately Positive"
    elif -0.1 < final_score < 0.1:
        sentiment = "Neutral"
    elif -0.5 <= final_score <= -0.1:
        sentiment = "Moderately Negative"
    else:
        sentiment = "Very Negative"

    # Get extracted details
    extracted = extract_details_with_gemini(text)
    extracted.update(extract_details_from_eml(msg))
    # Combined flat JSON response
    return jsonify({
        "sentiment": sentiment,
        "score": final_score,
        "customer_name": extracted.get("customer_name"),
        "order_id": extracted.get("order_id"),
        "feedback_category": extracted.get("feedback_category"),
        "subject": extracted.get("subject"),
        "customer_email": extracted.get("customer_email"),
        "date": extracted.get("date"),
        "feedback_summary": extracted.get("feedback_summary"),
        "action_needed": extracted.get("action_needed")
    })

if __name__ == "__main__":
    app3.run()