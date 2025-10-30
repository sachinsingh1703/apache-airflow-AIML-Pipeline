import pandas as pd
from faker import Faker
import datetime
import random
import os

# DO NOT import uuid

# 1. Define the output directory and ensure it exists.
OUTPUT_DIR = "/opt/airflow/data/generated_users"
BATCH_SIZE = 100000

# 2. Initialize Faker
fake = Faker()

def main():
    """
    Main function to generate and save a multi-table dataset with PK/FK relationships.
    """
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory '{OUTPUT_DIR}' is ready.")

    # Define date ranges for data generation
    two_years_ago = datetime.datetime.now() - datetime.timedelta(days=2 * 365)
    one_year_ago = datetime.datetime.now() - datetime.timedelta(days=365)
    now = datetime.datetime.now()

    # --- GENERATE DIMENSION TABLES FIRST ---
    # The generation order is stores -> products -> customers to follow the "fewer rows first" rule.
    # All dimension tables must be generated before the fact table (sales) that depends on them.

    # Table: stores (Small table, no batching needed)
    print("\nGenerating 'stores' table...")
    stores_rows = 1000
    stores_data = []
    store_id_list = []
    for i in range(1, stores_rows + 1):
        store_id = f"S{i:03d}"
        store_id_list.append(store_id)
        stores_data.append({
            "store_id": store_id,
            "store_name": fake.company(),
            "city": fake.city(),
            "state": fake.state_abbr(),
        })
    stores_df = pd.DataFrame(stores_data)
    stores_df.to_parquet(os.path.join(OUTPUT_DIR, 'stores.parquet'), index=False)
    print(f"-> Saved 'stores.parquet' with {len(stores_df)} rows.")

    # Table: products (Large table, requires batching)
    print("\nGenerating 'products' table (batched)...")
    products_rows = 900000
    product_id_list = []
    product_dfs = []
    product_batch_data = []
    for i in range(1, products_rows + 1):
        product_id = f"PROD{i+10000}" # Start from 10001
        product_id_list.append(product_id)
        product_batch_data.append({
            "product_id": product_id,
            "product_name": fake.catch_phrase(),
            "category": fake.bs(),
            "unit_price": round(random.uniform(5.0, 150.0), 2),
        })
        if i % BATCH_SIZE == 0 or i == products_rows:
            print(f"  ...processing batch ending at row {i}")
            product_dfs.append(pd.DataFrame(product_batch_data))
            product_batch_data = []

    products_df = pd.concat(product_dfs, ignore_index=True)
    products_df.to_parquet(os.path.join(OUTPUT_DIR, 'products.parquet'), index=False)
    print(f"-> Saved 'products.parquet' with {len(products_df)} rows.")

    # Table: customers (Large table, requires batching)
    print("\nGenerating 'customers' table (batched)...")
    customers_rows = 1000000
    customer_id_list = []
    customer_dfs = []
    customer_batch_data = []
    for i in range(1, customers_rows + 1):
        customer_id = f"CUST-{i:04d}"
        customer_id_list.append(customer_id)
        customer_batch_data.append({
            "customer_id": customer_id,
            "customer_name": fake.name(),
            "email_address": fake.email(),
            "signup_date": fake.date_time_between(start_date=two_years_ago, end_date=now),
        })
        if i % BATCH_SIZE == 0 or i == customers_rows:
            print(f"  ...processing batch ending at row {i}")
            customer_dfs.append(pd.DataFrame(customer_batch_data))
            customer_batch_data = []

    customers_df = pd.concat(customer_dfs, ignore_index=True)
    customers_df.to_parquet(os.path.join(OUTPUT_DIR, 'customers.parquet'), index=False)
    print(f"-> Saved 'customers.parquet' with {len(customers_df)} rows.")

    # --- GENERATE FACT TABLE LAST ---
    # This table uses the primary key lists generated from the dimension tables.

    # Table: sales (Large table, requires batching)
    print("\nGenerating 'sales' table (batched)...")
    sales_rows = 700000
    sales_dfs = []
    sales_batch_data = []
    for i in range(1, sales_rows + 1):
        sales_batch_data.append({
            "sale_id": f"SALE-{i:07d}",
            "customer_id": random.choice(customer_id_list),
            "product_id": random.choice(product_id_list),
            "store_id": random.choice(store_id_list),
            "quantity_sold": random.randint(1, 5),
            "transaction_date": fake.date_time_between(start_date=one_year_ago, end_date=now),
        })
        if i % BATCH_SIZE == 0 or i == sales_rows:
            print(f"  ...processing batch ending at row {i}")
            sales_dfs.append(pd.DataFrame(sales_batch_data))
            sales_batch_data = []
            
    sales_df = pd.concat(sales_dfs, ignore_index=True)
    sales_df.to_parquet(os.path.join(OUTPUT_DIR, 'sales.parquet'), index=False)
    print(f"-> Saved 'sales.parquet' with {len(sales_df)} rows.")

    print("\nAll data generation complete.")


if __name__ == "__main__":
    main()