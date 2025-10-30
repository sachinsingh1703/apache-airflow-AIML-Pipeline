import streamlit as st
import pandas as pd
import glob
import os
import google.generativeai as genai
import re
import requests
from requests.auth import HTTPBasicAuth
import time
import json
import zipfile  
import io       

# --- Configuration ---
GENERATOR_FILE_PATH = "dags/utils/database_generator.py"
DATA_DIR = "data/generated_users"
DAG_ID = "ai_database_generator" # Make sure this matches your DAG's dag_id

# --- Airflow API Configuration ---
AIRFLOW_API_URL = os.environ.get("AIRFLOW_API_URL", "http://airflow-webserver:8080/api/v1")
AIRFLOW_USER = os.environ.get("AIRFLOW_USER", "airflow")
AIRFLOW_PASS = os.environ.get("AIRFLOW_PASS", "airflow")
AIRFLOW_AUTH = HTTPBasicAuth(AIRFLOW_USER, AIRFLOW_PASS)

st.set_page_config(layout="wide", page_title="AI Database Generator")
st.title("🤖 AI Multi-Table Database Generator")

# --- Helper Functions ---

def load_generator_code():
    if not os.path.exists(GENERATOR_FILE_PATH):
        return "# Please define a schema and generate code."
    with open(GENERATOR_FILE_PATH, "r") as f:
        return f.read()

def save_generator_code(code_text):
    try:
        with open(GENERATOR_FILE_PATH, "w") as f:
            f.write(code_text)
        return True
    except Exception as e:
        st.error(f"Failed to save file: {e}")
        return False

def clean_gemini_response(text):
    text = text.replace("```python", "").replace("```", "")
    return text.strip()

def create_zip_archive(parquet_files):
    """
    Reads multiple .parquet files, converts each to CSV,
    and returns an in-memory ZIP file.
    """
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_f:
            total_files = len(parquet_files)
            progress_bar = st.progress(0, text="Zipping files...")
            
            for i, file_path in enumerate(parquet_files):
                df = pd.read_parquet(file_path)
                csv_data = df.to_csv(index=False)
                base_name = os.path.basename(file_path)
                csv_file_name = base_name.replace('.parquet', '.csv')
                zip_f.writestr(csv_file_name, csv_data)
                
                progress_text = f"Zipping {csv_file_name}... ({i+1}/{total_files})"
                progress_bar.progress((i + 1) / total_files, text=progress_text)
        
        progress_bar.empty()
        zip_buffer.seek(0)
        return zip_buffer
    except Exception as e:
        st.error(f"Failed to create zip file: {e}")
        progress_bar.empty()
        return None

# --- Airflow API Functions ---

def trigger_airflow_dag():
    url = f"{AIRFLOW_API_URL}/dags/{DAG_ID}/dagRuns"
    headers = {"Content-Type": "application/json"}
    body = {"conf": {}}
    try:
        response = requests.post(url, auth=AIRFLOW_AUTH, headers=headers, json=body)
        response.raise_for_status() 
        data = response.json()
        dag_run_id = data.get("dag_run_id")
        st.session_state.dag_run_id = dag_run_id
        return dag_run_id
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to trigger DAG. Is Airflow running? Error: {e}")
        return None

def get_dag_run_status():
    if "dag_run_id" not in st.session_state:
        return None
    dag_run_id = st.session_state.dag_run_id
    url = f"{AIRFLOW_API_URL}/dags/{DAG_ID}/dagRuns/{dag_run_id}"
    try:
        response = requests.get(url, auth=AIRFLOW_AUTH)
        response.raise_for_status()
        data = response.json()
        return data.get("state") 
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get DAG status: {e}")
        return "failed" 

# --- Gemini Code Generation Function (UPDATED PROMPT) ---

def call_gemini_api(schema, api_key):
    try:
        genai.configure(api_key=api_key)
        # Using the model you specified
        model = genai.GenerativeModel('gemini-2.5-pro') 
        
        schema_str = json.dumps(schema, indent=2)
        
        # --- PROMPT UPDATED to handle dynamic row counts ---
        full_prompt = f"""
        You are an expert Python data engineer. Your task is to write a single Python script to generate a multi-table dataset with specific Primary Key (PK) and Foreign Key (FK) requirements.

        You will be given a JSON object describing the tables, their relationships, and the number of rows for each.
        
        YOUR GOAL is to write a Python script with a single `main()` function. This script must:
        1.  Import necessary libraries: `pandas as pd`, `faker`, `datetime`, `random`, `os`. **DO NOT import uuid.**
        2.  Define the output directory: `OUTPUT_DIR = "/opt/airflow/data/generated_users"` and ensure it exists.
        3.  Initialize Faker: `fake = Faker()`.
        4.  **Generation Order:** Generate tables with fewer rows first ("Dimension").
        5.  **PATTERNED PRIMARY KEYS:** For columns specified as a Primary Key (PK) in the schema (e.g., 'customer_id'), check the user's prompt for that table. 
            * If the prompt mentions a specific pattern (like 'CUST-XXXX' or 'ORD followed by digits'), generate IDs matching that pattern using Python f-strings (e.g., `f"CUST-{{i:04d}}"` where `i` is the sequence number starting from 1).
            * If no pattern is mentioned, generate **sequential integers starting from 1**.
        6.  Store these generated PKs (whether patterned strings or integers) in a list in memory (e.g., `customer_id_list`).
        7.  Generate tables with many rows last ("Fact").
        8.  **FOREIGN KEYS:** When generating a Fact table's Foreign Key (FK) column (e.g., 'customer_id' in Sales), you MUST use `random.choice(customer_id_list)` to select a valid PK (string or integer) from the corresponding PK list. This ensures referential integrity.
        9.  **Row Counts:** Generate the exact number of rows specified for each table.
        10. **Batching:** For large tables (> 100,000 rows), generate data in batches.
        11. Save each table as a separate `.parquet` file in the `OUTPUT_DIR`.
        12. Include print statements for progress.

        **CRITICAL FAKER RULES (No UUIDs):**
        * For a **product name**: use `fake.catch_phrase()` or `fake.bs()`.
        * For **IDs NOT specified as PK/FK**: use `random.randint(1000, 9999)` or similar, but NOT patterned or sequential.
        * For **dates**: use `fake.date_time_between()`.
        
        Respond ONLY with the complete, runnable Python code. Do not include any markdown or explanation.

        ---
        HERE IS THE DATABASE SCHEMA:
        {schema_str}
        ---
        """
        
        response = model.generate_content(full_prompt)
        cleaned_response = response.text.replace("```python", "").replace("```", "").strip()
        return cleaned_response
    
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return None

# --- Streamlit UI ---

API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize state
if 'num_tables' not in st.session_state:
    st.session_state.num_tables = 1
    st.session_state.tables = {}
    st.session_state.current_code = load_generator_code()
    st.session_state.key_counter = 0
if 'code_is_saved' not in st.session_state:
    st.session_state.code_is_saved = os.path.exists(GENERATOR_FILE_PATH)

st.header("1. 🤖 Define Your Database Schema")

# --- Step 1: Define Number of Tables ---
num_tables = st.number_input("How many tables do you want to generate?", min_value=1, max_value=10, value=st.session_state.num_tables)
if num_tables != st.session_state.num_tables:
    st.session_state.num_tables = num_tables
    st.session_state.tables = {} 

# --- Step 2: Define Each Table ---
st.info("Define your small tables first. Large tables (like 1,000,000 rows) can link to them.")

table_definitions = {}
pk_options = []

# First pass to get names and PKs
for i in range(st.session_state.num_tables):
    table_key = f"table_{i}"
    # --- UPDATED: Default 'rows' instead of 'type' ---
    if table_key not in st.session_state.tables:
        st.session_state.tables[table_key] = {
            "name": f"Table{i+1}",
            "rows": 1000, # Default to 1000 rows
            "prompt": "",
            "pk": "",
            "fk": []
        }
    
    with st.expander(f"Table {i+1} Definition", expanded=True):
        t_def = st.session_state.tables[table_key]
        t_def['name'] = st.text_input(f"Table Name", value=t_def['name'], key=f"name_{i}")
        
        # --- UPDATED: Replaced radio with number_input ---
        t_def['rows'] = st.number_input(
            f"Number of Rows for {t_def['name']}",
            min_value=1,
            max_value=1_000_000,
            value=t_def.get('rows', 1000), # Use .get for safety
            key=f"rows_{i}"
        )
        st.caption("Max: 1,000,000. Use < 50,000 for 'Dimension' tables, > 50,000 for 'Fact' tables.")
        
        t_def['prompt'] = st.text_area(f"Prompt for {t_def['name']}", value=t_def['prompt'], key=f"prompt_{i}", placeholder=f"e.g., A student with a student_id, name, and email")
        t_def['pk'] = st.text_input(f"Primary Key Column Name", value=t_def['pk'], key=f"pk_{i}", placeholder=f"e.g., student_id")
        
        if t_def['name'] and t_def['pk']:
            pk_options.append(f"{t_def['name']}.{t_def['pk']}")

# Second pass to assign Foreign Keys
for i in range(st.session_state.num_tables):
    table_key = f"table_{i}"
    t_def = st.session_state.tables[table_key]
    with st.expander(f"Table {i+1} Links (Foreign Keys)"):
        available_links = [opt for opt in pk_options if not opt.startswith(f"{t_def['name']}.")]
        if available_links:
            t_def['fk'] = st.multiselect(
                f"Link {t_def['name']} to other tables:",
                options=available_links,
                default=t_def.get('fk', []),
                key=f"fk_{i}"
            )
        else:
            st.caption("Define other tables' Primary Keys to link them here.")

# --- Step 3: Generate Code ---
st.markdown("---")
st.header("2. 🤖 Generate & Save Code")

if st.button("Generate Database Code", use_container_width=True, type="primary"):
    with st.spinner("Calling Gemini API..."):
        generated_code = call_gemini_api(st.session_state.tables, API_KEY)
    if generated_code:
        st.session_state.current_code = generated_code
        st.session_state.key_counter += 1
        st.session_state.code_is_saved = False
        st.success("Code generated! Review and save below.")
    else:
        st.error("AI failed to generate code.")

# Editor to show generated code
editor_key = f"code_editor_{st.session_state.key_counter}"
code_in_editor = st.text_area(
    "AI-Generated Code (Edit as needed):", 
    value=st.session_state.current_code, 
    height=400, 
    key=editor_key
)

if st.button("💾 Save Code to Airflow", use_container_width=True):
    code_to_save = st.session_state[editor_key]
    save_generator_code(code_to_save)
    st.session_state.current_code = code_to_save
    st.session_state.code_is_saved = True
    st.success("Code saved to file!")

# --- Step 4: Run Generation ---
st.markdown("---")
st.header("3. 🚀 Run Pipeline & Download Data")

if not st.session_state.get("code_is_saved", False):
    st.warning("Please **Save Code** before starting data generation.")
else:
    if st.button("🚀 Start Database Generation", type="primary", use_container_width=True):
        if st.session_state[editor_key] != load_generator_code():
             st.warning("Your latest edits are not saved. Please click 'Save Code' first.")
        else:
            dag_run_id = trigger_airflow_dag()
            if dag_run_id:
                st.session_state.monitoring_dag = True
                st.info(f"Successfully triggered Airflow DAG run: `{dag_run_id}`")
            else:
                st.error("Failed to trigger DAG. Check the Airflow Webserver logs.")

# Polling logic
if st.session_state.get("monitoring_dag", False):
    with st.spinner("Data generation in progress... This may take several minutes. Polling Airflow every 10 seconds."):
        status = "running"
        while status == "running":
            time.sleep(10) 
            status = get_dag_run_status()
        
        if status == "success":
            st.success("Data generation complete! ✅")
            st.balloons()
            st.session_state.monitoring_dag = False
            st.rerun() # Rerun to show files
            
        elif status == "failed":
            st.error(f"DAG run {st.session_state.dag_run_id} failed. Please check the Airflow UI for logs.")
            st.session_state.monitoring_dag = False

# --- Step 5: Validate & Download (with .zip) ---
st.subheader("Generated Database Files")
st.info(f"Files are saved as Parquet in your project's `{DATA_DIR}` folder.")

parquet_files = sorted(glob.glob(f"{DATA_DIR}/*.parquet"))

if not parquet_files:
    st.warning("No data files found. Please run your Airflow DAG first.")
else:
    st.success(f"Found {len(parquet_files)} database tables!")
    
    for f in parquet_files:
        st.markdown(f"- `{os.path.basename(f)}`")
        
    st.subheader("Preview First Table")
    try:
        sample_df = pd.read_parquet(parquet_files[0])
        st.dataframe(sample_df.head(100), hide_index=True)
    except Exception as e:
        st.error(f"Failed to read sample file {parquet_files[0]}: {e}")

    st.subheader("Download All Tables (.zip)")
    if st.button("📦 Prepare All Tables as .zip", type="primary", use_container_width=True):
        zip_data = create_zip_archive(parquet_files)
        if zip_data:
            st.session_state.zip_data_ready = True
            st.session_state.zip_data = zip_data
            st.success("Zip file is ready to download!")
    
    if st.session_state.get("zip_data_ready", False):
        st.download_button(
            label="⬇️ Download database.zip",
            data=st.session_state.zip_data,
            file_name="generated_database.zip",
            mime="application/zip",
            use_container_width=True
        )

    