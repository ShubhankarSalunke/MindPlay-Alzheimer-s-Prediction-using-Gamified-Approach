import xgboost as xgb
import utils

X_train, X_test, y_train, y_test = utils.load_and_preprocess_data()

model = xgb.XGBClassifier(objective="multi:softmax", num_class=3, n_estimators=100)

model.fit(X_train, y_train)

model.save_model("models/alzheimers_xgboost.json")

print("Model trained and saved successfully!")
