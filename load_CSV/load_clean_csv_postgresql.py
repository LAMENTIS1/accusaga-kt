import pandas as pd
import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import tempfile
import gc

# Automatically loads .env from current directory or parent dirs
load_dotenv()

def import_csv_to_postgres(csv_file, table_name, create_table=True):
    """
    Import a CSV file to PostgreSQL, optionally creating the table.
    
    Args:
        csv_file (str): Path to the CSV file.
        table_name (str): Name of the PostgreSQL table.
        create_table (bool): Whether to create the table (only for the first chunk).
    """
    
    # Read CSV in chunks to normalize
    chunk_size = 10000  # Adjust chunk size based on memory constraints
    chunks = pd.read_csv(csv_file, dtype=str, low_memory=False, chunksize=chunk_size)

    # Load DB credentials from .env
    db_name = os.getenv('POSTGRES_DB')
    db_user = os.getenv('POSTGRES_USER')
    db_password = os.getenv('POSTGRES_PASSWORD')
    db_host = os.getenv('POSTGRES_HOST', 'localhost')
    db_port = os.getenv('POSTGRES_PORT', '5432')

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )
    cur = conn.cursor()

    # Create table only for the first chunk
    if create_table:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name)))
        create_query = sql.SQL("""
            CREATE TABLE {} (
                ResponseId TEXT,
                attribute TEXT,
                value TEXT
            )
        """).format(sql.Identifier(table_name))
        cur.execute(create_query)
        conn.commit()

    # Process each chunk
    for i, chunk in enumerate(chunks):
        
        # Normalize the chunk (melt)
        df_long = pd.melt(chunk, id_vars=['ResponseId'], var_name='attribute', value_name='value')

        # Save normalized chunk to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_file_path = temp_file.name
            df_long.to_csv(temp_file_path, index=False)
        
        # Import the temporary CSV to PostgreSQL
        with open(temp_file_path, 'r') as f:
            next(f)  # Skip header
            cur.copy_expert(
                sql.SQL("COPY {} FROM STDIN WITH CSV HEADER").format(sql.Identifier(table_name)),
                f
            )
        conn.commit()

        # Clean up temporary file
        os.remove(temp_file_path)
        del df_long, chunk
        gc.collect()  # Force garbage collection to free memory

    cur.close()
    conn.close()

def load_clean_csv_to_postgresql(file_path, table_name, chunk_size=10000):
    """
    Load and clean a CSV or Excel file, then import to PostgreSQL in chunks.
    
    Args:
        file_path (str): Path to the input file (.csv, .xlsx, .xls).
        table_name (str): Name of the PostgreSQL table.
        chunk_size (int): Number of rows per chunk.
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    # Determine file type and prepare chunked reading
    if file_extension in ['.csv']:
        chunks = pd.read_csv(file_path, dtype=str, low_memory=False, chunksize=chunk_size)
    elif file_extension in ['.xlsx', '.xls']:
        # Excel files are trickier for chunking; load in chunks manually if possible
        chunks = pd.read_excel(file_path, dtype=str, chunksize=chunk_size)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}. Only .csv, .xlsx, or .xls are supported.")

    # Function to check if a string is JSON-like
    def is_json_like(s):
        if isinstance(s, str):
            s = s.strip()
            return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))
        return False

    # Process chunks
    first_chunk = True
    for i, chunk in enumerate(chunks):
        

        # Ensure 'ResponseId' exists
        if 'ResponseId' not in chunk.columns:
            chunk.insert(0, 'ResponseId', range(i * chunk_size + 1, i * chunk_size + len(chunk) + 1))
        else:
            print("'ResponseId' column exists.")

        # No column dropping — keep all original columns
        df_cleaned = chunk.copy()

        # Check for JSON-like content in rows 0 or 1 of the first chunk
        drop_first_two = False
        if first_chunk:
            check_rows = [0, 1]
            for idx in check_rows:
                if idx < len(df_cleaned):
                    row = df_cleaned.iloc[idx]
                    if any(is_json_like(str(cell)) for cell in row):
                        drop_first_two = True
                        break
            if drop_first_two:
                df_cleaned = df_cleaned.drop(index=[i for i in check_rows if i < len(df_cleaned)]).reset_index(drop=True)
            else:
                print("No JSON-like content in row 1 or 2. Keeping rows.")


        # Save cleaned chunk to a temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_file_path = temp_file.name
            df_cleaned.to_csv(temp_file_path, index=False)

        # Import the cleaned chunk to PostgreSQL
        import_csv_to_postgres(temp_file_path, table_name, create_table=first_chunk)

        # Clean up
        os.remove(temp_file_path)
        del df_cleaned, chunk
        gc.collect()  # Free memory
        first_chunk = False  # Only create table for the first chunk

    print("[DONE] File processing and import complete.")

