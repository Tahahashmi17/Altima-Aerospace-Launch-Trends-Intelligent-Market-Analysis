from src.data_preprocessing import load_and_clean_data, standardize_categories, categorize_description, final_cleanup
from src.eda import run_eda
from src.feature_engineering import engineer_features
from src.modeling import train_models
from src.evaluation import evaluate_model

import os
from joblib import dump

# Load and preprocess data
df = load_and_clean_data("data/Launch SFR.csv")
df = standardize_categories(df)
df = categorize_description(df)
df = final_cleanup(df)
run_eda(df)

# Feature engineering
df = engineer_features(df)

# Model training
rfc_model, dtc_model, X_test, y_test = train_models(df)
print("Models trained successfully.")

# Save models to 'models/' directory
os.makedirs("models", exist_ok=True)
dump(rfc_model, "models/random_forest.pkl")
dump(dtc_model, "models/decision_tree.pkl")
print("Models saved to 'models/' directory.")

# Evaluate models
evaluate_model(rfc_model, X_test, y_test, "Random Forest")
evaluate_model(dtc_model, X_test, y_test, "Decision Tree")
