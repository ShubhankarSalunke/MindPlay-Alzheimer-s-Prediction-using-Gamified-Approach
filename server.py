import xgboost as xgb
import numpy as np

def aggregate_models(client_models):
    """Aggregates models from multiple federated learning clients."""
    global_model = xgb.Booster()

    boosters = [xgb.Booster(model_file=model_path) for model_path in client_models]

    global_model = boosters[0]
    for booster in boosters[1:]:
        global_model = merge_boosters(global_model, booster)

    global_model.save_model("models/alzheimers_xgboost.json")
    print("Federated model aggregation complete!")

def merge_boosters(model1, model2):
    """Merge two XGBoost models by averaging their feature importance."""
    fscore1 = model1.get_fscore()
    fscore2 = model2.get_fscore()

    merged_fscore = {}
    all_features = set(fscore1.keys()).union(set(fscore2.keys()))
    for feature in all_features:
        merged_fscore[feature] = (fscore1.get(feature, 0) + fscore2.get(feature, 0)) / 2

    model1.set_attr(**{f"f{i}": str(v) for i, (k, v) in enumerate(merged_fscore.items())})
    
    return model1

aggregate_models(["clients/client_1_model.json", "clients/client_2_model.json"])
