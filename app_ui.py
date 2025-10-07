import tkinter as tk
from tkinter import ttk, messagebox
import os
import joblib

# We will reuse the core functions from our command-line script
from predict import find_latest_model, predict_fraud

class FraudDetectorApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Fraud Detection Tool")
        self.root.geometry("400x250") # Set window size

        # Load the model once on startup
        self.model = self.load_model()
        if not self.model:
            messagebox.showerror("Model Not Found", 
                                 "No trained model file (.pkl) found in 'data/models'.\n"
                                 "Please run the Airflow DAG to train a model first.")
            self.root.destroy()
            return
            
        self.create_widgets()

    def load_model(self):
        """Loads the latest model file from the specified directory."""
        latest_model_path = find_latest_model('data/models')
        if latest_model_path:
            print(f"📂 Loading model: {os.path.basename(latest_model_path)}")
            return joblib.load(latest_model_path)
        return None

    def create_widgets(self):
        """Creates and arranges all the UI elements in the window."""
        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Input Fields ---
        ttk.Label(main_frame, text="Transaction Type:").grid(column=0, row=0, sticky=tk.W, pady=5)
        self.ttype_var = tk.StringVar()
        transaction_types = ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'CASH_IN', 'DEBIT']
        type_dropdown = ttk.OptionMenu(main_frame, self.ttype_var, transaction_types[0], *transaction_types)
        type_dropdown.grid(column=1, row=0, sticky=tk.EW)

        ttk.Label(main_frame, text="Amount:").grid(column=0, row=1, sticky=tk.W, pady=5)
        self.amount_var = tk.StringVar(value="100000") # Default value
        ttk.Entry(main_frame, width=25, textvariable=self.amount_var).grid(column=1, row=1, sticky=tk.EW)

        ttk.Label(main_frame, text="Origin Balance (Before):").grid(column=0, row=2, sticky=tk.W, pady=5)
        self.oldbalanceOrg_var = tk.StringVar(value="100000") # Default value
        ttk.Entry(main_frame, width=25, textvariable=self.oldbalanceOrg_var).grid(column=1, row=2, sticky=tk.EW)

        # --- Predict Button ---
        predict_button = ttk.Button(main_frame, text="Predict Fraud", command=self.perform_prediction)
        predict_button.grid(column=1, row=3, pady=20)

        # --- Result Label ---
        self.result_var = tk.StringVar(value="Result will be shown here...")
        self.result_label = ttk.Label(main_frame, textvariable=self.result_var, font=("Helvetica", 12, "bold"))
        self.result_label.grid(column=0, row=4, columnspan=2, pady=10)

        # Configure column weights for resizing
        main_frame.columnconfigure(1, weight=1)

    def perform_prediction(self):
        """Gathers input, calls the prediction logic, and updates the UI."""
        # 1. Get and validate inputs
        try:
            amount_val = float(self.amount_var.get())
            oldbalanceOrg_val = float(self.oldbalanceOrg_var.get())
            ttype_val = self.ttype_var.get()
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for amount and balance.")
            return

        # 2. Construct the feature dictionary for the model
        newbalanceOrig_val = oldbalanceOrg_val - amount_val
        transaction_data = {
            'amount': amount_val,
            'oldbalanceOrg': oldbalanceOrg_val,
            'newbalanceOrig': newbalanceOrig_val,
            'oldbalanceDest': 0.0,
            'newbalanceDest': amount_val,
            'balance_delta_orig': -amount_val,
            'balance_delta_dest': amount_val,
            'emptied_account': 1 if newbalanceOrig_val == 0 else 0,
            'type': ttype_val
        }

        # 3. Call the imported prediction function
        result, confidence = predict_fraud(self.model, transaction_data)
        
        # 4. Update the result label with the outcome and color
        display_text = f"Prediction: {result} (Confidence: {confidence:.2%})"
        self.result_var.set(display_text)
        
        if result == "Fraudulent":
            self.result_label.config(foreground="red")
        else:
            self.result_label.config(foreground="green")

if __name__ == "__main__":
    app_root = tk.Tk()
    app = FraudDetectorApp(app_root)
    app_root.mainloop()