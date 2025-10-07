import joblib
import pandas as pd
import os
import glob
import warnings

# Suppress a harmless warning from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

def find_latest_model(model_dir: str):
    """Finds the most recently created model file in a directory."""
    list_of_files = glob.glob(os.path.join(model_dir, '*.pkl'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def predict_fraud(model, transaction_data: dict):
    """
    Predicts if a single transaction is fraudulent using a trained model.
    """
    # 1. Convert input dictionary to a pandas DataFrame
    df = pd.DataFrame([transaction_data])

    # 2. Perform the EXACT same one-hot encoding as in training
    # This is critical. The model expects the same columns it was trained on.
    df['type'] = pd.Categorical(
        df['type'],
        categories=['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    )
    df_processed = pd.get_dummies(df, columns=['type'], drop_first=True)

    # 3. Ensure all columns the model was trained on are present and in the correct order
    model_features = model.feature_names_in_
    df_processed = df_processed.reindex(columns=model_features, fill_value=0)

    # 4. Make prediction
    prediction = model.predict(df_processed)[0]
    probability = model.predict_proba(df_processed)[0]

    # 5. Return a human-readable result
    if prediction == 1:
        result = "Fraudulent"
        confidence = probability[1] # Probability of being '1' (Fraud)
    else:
        result = "Not Fraudulent"
        confidence = probability[0] # Probability of being '0' (Not Fraud)

    return result, confidence


if __name__ == "__main__":
    model_directory = 'data/models'
    latest_model_path = find_latest_model(model_directory)

    if not latest_model_path:
        print(f"❌ Error: No model files found in '{model_directory}'.")
        print("Please run the Airflow DAG to train and save a model first.")
    else:
        print(f"📂 Loading the latest model: {os.path.basename(latest_model_path)}\n")
        loaded_model = joblib.load(latest_model_path)

        # --- Get Transaction Input from User ---
        print("Enter transaction details (press Enter to accept default values):")

        # Set default values for a potentially fraudulent transaction
        amount = float(input("Amount [default: 100000.0]: ") or "100000")
        oldbalanceOrg = float(input("Origin Account Balance (Before) [default: 100000.0]: ") or "100000")
        ttype = input("Transaction Type (CASH_OUT, TRANSFER, etc.) [default: CASH_OUT]: ") or "CASH_OUT"

        # Construct the feature dictionary based on user input
        newbalanceOrig = oldbalanceOrg - amount # Assumes the full amount is transferred
        
        user_transaction = {
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': 0.0,
            'newbalanceDest': amount, # Assume destination starts at 0 and receives full amount
            'balance_delta_orig': -amount,
            'balance_delta_dest': amount,
            'emptied_account': 1 if newbalanceOrig == 0 else 0,
            'type': ttype
        }

        print("\n--- Running Prediction ---")
        result, confidence = predict_fraud(loaded_model, user_transaction)
        print(f"✅ Prediction: {result} (Confidence: {confidence:.2%})")