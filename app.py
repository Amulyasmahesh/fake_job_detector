import streamlit as st
import pickle
import re

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

# Prediction function
def predict_job(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    result = model.predict(vec)
    
    if result[0] == 1:
        return "🚨 Fake Job Posting"
    else:
        return "✅ Real Job Posting"

# UI
st.set_page_config(page_title="Fake Job Detector", page_icon="💼") 
st.title("💼 Fake Job Posting Detector")
st.write("Enter job description to check if it's fake or real")

user_input = st.text_area("Job Description")

if st.button("Predict"):
    if user_input.strip() != "":
        prediction = predict_job(user_input)
        st.subheader(prediction)
    else:
        st.warning("Please enter some text")