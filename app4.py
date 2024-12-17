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
import logging
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    """
    Use SpaCy for tokenization and lemmatization.
    """
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]

def is_customer_feedback(subject, text):
    """
    Check if the email is a customer feedback email based on subject and content.
    Returns: bool
    """
    # Keywords that indicate customer feedback
    feedback_keywords = [
        r'\bfeedback\b', r'\bcomplaint\b', r'\breview\b', r'\bexperience\b',
        r'\bsuggestion\b', r'\bissue\b', r'\bproblem\b', r'\bconcern\b',
        r'\bdissatisfied\b', r'\bsatisfied\b', r'\brating\b'
    ]

    # Keywords that indicate non-customer feedback emails
    exclude_keywords = [
        r'\border confirmation\b', r'\bshipping update\b', r'\bnewsletter\b',
        r'\bpromotion\b', r'\bdeal\b', r'\boffer\b', r'\bsubscription\b',
        r'\bpassword\b', r'\breceipt\b', r'\binvoice\b', r'\bmarketing\b', r'\badvertisement\b'
    ]

    # Preprocess the subject and text
    subject_tokens = preprocess(subject)
    text_tokens = preprocess(text)

    # Combine tokens back into strings for regex matching
    processed_subject = " ".join(subject_tokens)
    processed_text = " ".join(text_tokens)

    # Check for feedback keywords
    has_feedback_keywords = any(
        re.search(keyword, processed_subject) or re.search(keyword, processed_text)
        for keyword in feedback_keywords
    )

    # Check for exclusion keywords
    has_exclude_keywords = any(
        re.search(keyword, processed_subject) or re.search(keyword, processed_text)
        for keyword in exclude_keywords
    )

    # Prioritize feedback keywords over exclusion keywords
    if has_feedback_keywords and has_exclude_keywords:
        logging.info("Both feedback and exclusion keywords found. Prioritizing feedback.")
        return True

    # Log the decision
    logging.info(f"Subject: {subject}, Feedback: {has_feedback_keywords}, Exclude: {has_exclude_keywords}")

    return has_feedback_keywords and not has_exclude_keywords


app4 = Flask(__name__)

# Configure Gemini API with direct API key
GOOGLE_API_KEY = "AIzaSyAcxV7BCseOR4kSyebWIM2c3T9r8R7TDuc"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def extract_details_with_gemini(text):
    prompt = f"""
    Extract these details from the email text and return as JSON object:
    - customer_name
    - order_id (if available) else return "N/A"
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

@app4.route('/', methods=['POST'])
def analyze_sentiment():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file.filename.endswith('.eml'):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        msg = BytesParser(policy=policy.default).parse(file.stream)
        subject = msg.get('subject', '')

        # Extract text content from the email
        text = ""
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8")
                break  # Exit loop after finding plain text content

        # Check if this is a customer feedback email
        if not is_customer_feedback(subject, text):
            return jsonify({
                "error": "Not a customer feedback email",
                "subject": subject
            }), 400

        # Get sentiment using both TextBlob and VADER
        blob = TextBlob(text)
        tb_score = blob.sentiment.polarity

        analyzer = SentimentIntensityAnalyzer()
        vader_scores = analyzer.polarity_scores(text)
        compound_score = vader_scores['compound']

        final_score = (tb_score + compound_score) / 2
        sentiment = classify_sentiment(final_score)

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
    except Exception as e:
        logging.error(f"Error processing email: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500



def classify_sentiment(final_score):
    """
    Classifies the sentiment based on the final score.
    """
    if final_score >= 0.5:
        return "Very Positive"
    elif 0.1 <= final_score < 0.5:
        return "Moderately Positive"
    elif -0.1 < final_score < 0.1:
        return "Neutral"
    elif -0.5 <= final_score <= -0.1:
        return "Moderately Negative"
    else:
        return "Very Negative"
if __name__ == "__main__":
    app4.run()
