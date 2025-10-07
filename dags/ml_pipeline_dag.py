# /dags/ml_pipeline_dag.py
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Add the 'finance' package to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from finance.clean_data import clean_data_in_postgres
from finance.train_model import train_and_save_model
from finance.evaluate_model import evaluate_saved_model

with DAG(
    dag_id='fraud_detection_ml_pipeline',
    start_date=datetime(2023, 10, 26),
    schedule=None,
    catchup=False,
    doc_md="### End-to-End ML Pipeline for Fraud Detection",
    tags=['ml', 'finance', 'end-to-end'],
) as dag:
    
    clean_data_task = PythonOperator(
        task_id='clean_and_feature_engineer',
        python_callable=clean_data_in_postgres
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_and_save_model,
        op_kwargs={'run_id': '{{ run_id }}'}
    )
    
    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_saved_model,
        op_kwargs={'model_path': "{{ task_instance.xcom_pull(task_ids='train_model') }}"}
    )

    # Define the task dependencies
    clean_data_task >> train_model_task >> evaluate_model_task