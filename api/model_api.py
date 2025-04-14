from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import numpy as np
import utils

app = Flask(__name__)

# Load trained model
model = xgb.XGBClassifier()
model.load_model("models/alzheimers_xgboost.json")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame(data)

    # Preprocess input
    X = df.drop(columns=["PatientID"], errors="ignore")
    X["Gender"] = X["Gender"].map({"Male": 0, "Female": 1})

    # Make predictions
    preds = model.predict(X)

    return jsonify({"predictions": preds.tolist()})

if __name__ == "__main__":
    app.run(port=5000)
