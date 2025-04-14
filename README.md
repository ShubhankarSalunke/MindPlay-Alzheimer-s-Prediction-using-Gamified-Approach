# 🧠 MindPlay: Gamified Cognitive Monitoring and Alzheimer's Prediction

MindPlay is an AI-powered, interactive health monitoring platform that combines cognitive games with machine learning to assess users' cognitive performance and predict early signs of Alzheimer's. It is designed to provide a personalized and engaging approach to brain health using Explainable and Federated AI techniques.

---

## 🚀 Features

- 🎮 **Gamified Cognitive Tests**: A suite of mini-games that test memory, attention, and problem-solving skills.
- 🤖 **Alzheimer’s Risk Prediction**: Machine learning models analyze gameplay data to estimate potential cognitive decline.
- 🔐 **Federated Learning**: Ensures user privacy by training models locally on user data without sharing raw information.
- 📊 **Personalized Reports**: Real-time feedback and progress reports tailored to each user's performance.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python-based UI)
- **Backend**: Python
- **Machine Learning**: Scikit-learn, XGBoost
- **Federated Learning**: Custom simulation of federated model updates
- **Data Storage**: Local CSV
- **Additional Tools**: Matplotlib, Pandas, NumPy

---

## 📈 Model Performance

- **Accuracy**: ~89% on synthetic cognitive test dataset
- **Improvements**: 25% boost in user engagement compared to static cognitive test tools
- **Explainability**: Feature contributions visible for every prediction

---

## 🧪 Cognitive Games Included

| Game Name      | Skill Tested          | Description                                       |
|----------------|------------------------|---------------------------------------------------|
| Memory Matrix  | Short-term memory      | Tap on an increasing sequence of tiles            |
| Color Switch   | Attention & flexibility| Match the word color, not the word itself         |
| Pattern Recall | Working memory         | Recreate shown visual sequences after delay       |
| Symbol Match   | Speed & recognition    | Match symbols under time pressure                 |

---

## 🧬 Alzheimer's Detection Pipeline

1. **Game Interaction** →  
2. **Feature Extraction (gameplay data)** →  
3. **Preprocessing & Normalization** →  
4. **Model Inference (XGBoost)** →  
5. **Explanation Layer (SHAP)** →  
6. **User Report**

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mindplay.git
cd mindplay

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
