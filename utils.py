import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    df = df.drop(columns=["PatientID", "Alzheimer's Status"])

    encoder = LabelEncoder()
    cols_to_encode = ['Gender', 'Digit Span Correct', 'Arithmetic Correct', 'Symbol Matching Correct', "Alzheimer's Binary"]

    for cols in cols_to_encode:
        df[cols] = encoder.fit_transform(df[cols])

    # Define features and target variable
    X = df.drop(columns=["Alzheimer's Binary"])
    y = df["Alzheimer's Binary"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

