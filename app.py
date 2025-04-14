import streamlit as st
import xgboost as xgb
import numpy as np

def load_model():
    model = xgb.XGBClassifier()
    model.load_model("models/alzheimers_xgboost.json")
    return model

model = load_model()

def predict(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]

    if prediction == 0:
        return "🟢 Low Risk (No Alzheimer's)"
    elif prediction == 1:
        return "🟡 Moderate Risk (At-Risk)"
    else:
        return "🔴 High Risk (Alzheimer’s)"

st.title("🧠 Federated Alzheimer's Risk Prediction")

age = st.slider("Age", 50, 85, 65)
gender = st.radio("Gender", ["Male", "Female"])

reaction_time_arithmetic = st.slider("Reaction Time (Arithmetic, ms)", 400, 2500, 1000)
reaction_time_word_recall = st.slider("Reaction Time (Word Recall, ms)", 400, 2500, 1000)
reaction_time_symbol_matching = st.slider("Reaction Time (Symbol Matching, ms)", 400, 2500, 1000)

final_reaction_time = np.mean([reaction_time_arithmetic, reaction_time_word_recall, reaction_time_symbol_matching])

digit_span_correct = st.radio("Digit Span Test Correct?", ["Yes", "No"])
word_recall_score = st.slider("Word Recall Score (0-5)", 0, 5, 3)
arithmetic_correct = st.radio("Arithmetic Test Correct?", ["Yes", "No"])
symbol_matching_correct = st.radio("Symbol Matching Test Correct?", ["Yes", "No"])

gender_encoded = 0 if gender == "Male" else 1
digit_span_encoded = 1 if digit_span_correct == "Yes" else 0
arithmetic_encoded = 1 if arithmetic_correct == "Yes" else 0
symbol_matching_encoded = 1 if symbol_matching_correct == "Yes" else 0

features = [
    age,
    gender_encoded,
    reaction_time_arithmetic,
    reaction_time_word_recall,
    reaction_time_symbol_matching,
    final_reaction_time,
    digit_span_encoded,
    word_recall_score,
    arithmetic_encoded,
    symbol_matching_encoded
]

if st.button("🔍 Predict"):
    result = predict(features)
    st.subheader(f"🧠 Prediction: {result}")
