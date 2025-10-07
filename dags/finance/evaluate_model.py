import pandas as pd
import logging
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from airflow.providers.postgres.hooks.postgres import PostgresHook

def evaluate_saved_model(model_path: str):
    if not model_path or not os.path.exists(model_path):
        logging.error(f"❌ Model file not found at {model_path}. Skipping evaluation.")
        return

    logging.info(f"📂 Loading model from: {model_path}...")
    model = joblib.load(model_path)
    
    pg_hook = PostgresHook(postgres_conn_id='postgres_finance_db')
    engine = pg_hook.get_sqlalchemy_engine()
    
    df = pd.read_sql("SELECT * FROM cleaned_transactions;", engine)

    # --- START OF FIX ---
    # Apply the SAME one-hot encoding as in the training script
    logging.info("Applying one-hot encoding for evaluation...")
    df_processed = pd.get_dummies(df, columns=['type'], drop_first=True)
    # --- END OF FIX ---
    
    # Recreate the exact same test set from the PROCESSED dataframe
    target = 'isfraud'
    features_to_drop = [target, 'isflaggedfraud', 'type_fraud_combo']
    features = [col for col in df_processed.columns if col not in features_to_drop]

    X = df_processed[features]
    y = df_processed[target]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logging.info("📊 Evaluating model performance...")
    
    # Ensure test set columns match model's expected columns
    model_features = model.feature_names_in_
    X_test = X_test.reindex(columns=model_features, fill_value=0)

    predictions = model.predict(X_test)
    
    report = classification_report(y_test, predictions, target_names=['Not Fraud', 'Fraud'])
    
    logging.info("\n--- Classification Report ---\n" + report)