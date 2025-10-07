import logging
from airflow.providers.postgres.hooks.postgres import PostgresHook

def clean_data_in_postgres():
    """
    Takes a random sample from the raw data, cleans it, engineers features,
    and stores the result in 'cleaned_transactions'.
    """
    pg_hook = PostgresHook(postgres_conn_id='postgres_finance_db')
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    logging.info("🧹 Starting data cleaning and feature engineering on a 200k sample...")

    # This query now includes a Common Table Expression (CTE) to sample the data first
    cleaning_sql = """
    DROP TABLE IF EXISTS cleaned_transactions;

    CREATE TABLE cleaned_transactions AS
    WITH sampled_transactions AS (
        -- Select a random sample of 200,000 rows to avoid memory issues
        SELECT * FROM transaction ORDER BY RANDOM() LIMIT 200000
    )
    SELECT
        -- Retain important original columns
        type,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
        isFraud,
        isFlaggedFraud,

        -- Feature Engineering
        (newbalanceOrig - oldbalanceOrg) AS balance_delta_orig,
        (newbalanceDest - oldbalanceDest) AS balance_delta_dest,
        CASE WHEN newbalanceOrig = 0 AND oldbalanceOrg > 0 THEN 1 ELSE 0 END AS emptied_account,
        CONCAT(type, '_', isFraud) as type_fraud_combo
    FROM
        sampled_transactions
    WHERE
        amount > 0;
    """
    
    try:
        cursor.execute(cleaning_sql)
        conn.commit()
        logging.info("✅ 'cleaned_transactions' table created successfully from sample data.")
    except Exception as e:
        logging.error(f"❌ Error during data cleaning: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()