from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


try:
    qa_data = pd.read_csv(os.path.join(BASE_DIR, "career_guidance_qa_dataset_v2.csv"))
except FileNotFoundError:
    raise RuntimeError("career_guidance_qa_dataset_v2.csv not found. Place it in the same folder as app.py.")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text

questions = [preprocess(q) for q in qa_data['question']]
answers   = qa_data['answer'].tolist()
 
vectorizer       = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)


try:
    with open(os.path.join(BASE_DIR, "career_model.pkl"), "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("career_model.pkl not found. Train and save your model first.")

career_info = {
    "Data Scientist":         "Works with data to extract insights; requires Python, statistics, and ML skills. Focus on building models and communicating findings to stakeholders.",
    "AI Engineer":            "Builds and deploys AI/ML systems in production; needs deep learning expertise, MLOps knowledge, and strong software engineering practices.",
    "Software Developer":     "Designs and builds applications; requires programming, problem-solving, and knowledge of software development lifecycle and version control.",
    "Business Analyst":       "Bridges business and technology by gathering requirements, modeling processes, and recommending solutions; strong communication and analytical skills essential.",
    "HR Manager":             "Manages recruitment, employee relations, and organizational development; requires empathy, communication, and knowledge of labour laws and HR systems.",
    "Marketing":              "Plans and executes campaigns to attract and retain customers; requires creativity, data analysis, and expertise in digital marketing channels.",
    "UI/UX Designer":         "Designs intuitive user interfaces and experiences; requires creativity, empathy, user research skills, and proficiency in tools like Figma.",
    "Cyber Security Analyst": "Protects systems and data from threats; requires knowledge of networking, ethical hacking, security tools, and strong analytical thinking.",
}


@app.route('/')
def home():
    return render_template("index.html",
        confidence     = "N/A",
        confidence_pct = 0,
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        interest_ai       = int(request.form.get('interest_ai', 0))
        interest_business = int(request.form.get('interest_business', 0))
        interest_design   = int(request.form.get('interest_design', 0))

        data = pd.DataFrame([[
            float(request.form['programming']),
            float(request.form['communication']),
            float(request.form['creativity']),
            float(request.form['analytical']),
            float(request.form['leadership']),
            interest_ai,
            interest_business,
            interest_design,
        ]], columns=[
            "programming", "communication", "creativity", "analytical",
            "leadership", "interest_ai", "interest_business", "interest_design",
        ])

        prediction = model.predict(data)[0]

        if hasattr(model, "predict_proba"):
            confidence     = round(max(model.predict_proba(data)[0]) * 100, 2)
            confidence_pct = confidence   # plain number, e.g. 87.5 — HTML adds the %
        else:
            confidence     = "N/A"
            confidence_pct = 0

        info = career_info.get(prediction, "No description available.")

        return render_template("index.html",
            prediction     = prediction,
            confidence     = confidence,
            confidence_pct = confidence_pct,
            info           = info,
        )

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return render_template("index.html",
            prediction     = "Something went wrong. Please check your inputs and try again.",
            confidence     = "N/A",
            confidence_pct = 0,
            info           = "",
        )


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.form.get("message", "").strip()

        if not user_input:
            return jsonify({"response": "Please type a question first."})

        processed  = preprocess(user_input)
        user_vec   = vectorizer.transform([processed])
        sim        = cosine_similarity(user_vec, question_vectors)

        idx   = sim.argmax()
        score = sim[0][idx]

        if score < 0.25:
            return jsonify({
                "response": (
                    "I'm not sure about that. Try asking about a specific career, "
                    "skill, or roadmap — for example: "
                    "'How do I become a data scientist?' or 'Steps to learn cybersecurity'."
                )
            })

        return jsonify({"response": answers[idx]})

    except Exception as e:
        app.logger.error(f"Chat error: {e}")
        return jsonify({"response": "Something went wrong. Please try again."})


if __name__ == "__main__":
    app.run(debug=True)

 

