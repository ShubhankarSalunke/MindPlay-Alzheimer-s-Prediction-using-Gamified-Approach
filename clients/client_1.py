import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # encoder = LabelEncoder()
    # cols_to_encode = ['Gender', 'Digit Span Correct', 'Arithmetic Correct', 'Symbol Matching Correct', "Alzheimer's Binary"]

    # for cols in cols_to_encode:
    #     df[cols] = encoder.fit_transform(df[cols])

    X = df.drop(columns=["Alzheimer's Binary"])
    y = df["Alzheimer's Binary"]

    return train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = load_and_preprocess_data("C:\SHUBH 2022-2026\MIT WPU\TYBTECH\SEM 2\ML\Mini Project\Mini final\Codebase\data\cleaned_data.csv")

local_model = xgb.XGBClassifier(objective="multi:softmax", num_class=3, n_estimators=100)
local_model.fit(X_train, y_train)

local_model.save_model("client_1_model.json")
print("Local model trained and saved!")
