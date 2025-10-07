import pandas as pd
import logging
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from airflow.providers.postgres.hooks.postgres import PostgresHook

def train_and_save_model(run_id: str):
    pg_hook = PostgresHook(postgres_conn_id='postgres_finance_db')
    engine = pg_hook.get_sqlalchemy_engine()
    
    logging.info("Fetching cleaned data for model training...")
    df = pd.read_sql("SELECT * FROM cleaned_transactions;", engine)

    if df.empty:
        logging.warning("⚠️ Clean data table is empty. Skipping model training.")
        return None

    # --- START OF FIX ---
    # Convert categorical variables to numeric using one-hot encoding
    logging.info("Applying one-hot encoding to categorical features...")
    df_processed = pd.get_dummies(df, columns=['type'], drop_first=True)
    # --- END OF FIX ---

    # Define features and target from the PROCESSED dataframe
    target = 'isfraud'
    # Drop non-feature columns
    features_to_drop = [target, 'isflaggedfraud', 'type_fraud_combo']
    features = [col for col in df_processed.columns if col not in features_to_drop]
    
    X = df_processed[features]
    y = df_processed[target]

    # Split data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    logging.info("🤖 Training the fraud detection model...")
    model.fit(X_train, y_train)
    logging.info("✅ Model training complete.")
    
    # Save the model
    model_dir = "/opt/airflow/data/models"
    os.makedirs(model_dir, exist_ok=True)
    
    safe_run_id = run_id.replace(':', '-').replace('+', '_')
    model_filename = f"fraud_detection_model_{safe_run_id}.pkl"
    model_path = os.path.join(model_dir, model_filename)

    joblib.dump(model, model_path)
    logging.info(f"💾 Model saved to: {model_path}")

    return model_path