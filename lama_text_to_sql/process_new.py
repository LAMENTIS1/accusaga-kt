import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OpenAI API key")

client = OpenAI(api_key=openai_api_key)

def generate_keyword(values_preview, col):
    prompt = (
        f"""You are analyzing a survey dataset to generate descriptive summaries of each column, optimized for semantic search and column name prediction.
    
        Given a column name and Column values from that column, write a clear and concise description that captures the unique meaning of the data in everyday terms.

        Guidelines:
        - Do NOT start with generic phrases like "This column represents".
        - Do NOT repeat the column name.
        - Clearly convey what the column tracks or measures, using context from the sample values.
        - Be specific enough that someone could match this description to a question (e.g., "How often do people exercise?" → frequency of physical activity).
        - Highlight the key subject (e.g., age, income, diagnosis, job title, brand preference, satisfaction rating).
        - Keep the description short (3-4 sentences), natural-sounding, and easy to understand.
        - Avoid technical, programming, or SQL terminology.

        Note:
        - Add All Column values in a single line of Description.

        Column name: {col}  
        Column values: {values_preview}

        Description:"""
            )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

def describe_column(col, sample_vals):
    values_preview = f"Column capturing values such as: {', '.join(map(str, sample_vals))}"
    description = generate_keyword(values_preview, col)
    return {'Code': col, 'Description': description}

def process_file_new(input_file, chunk_size=10000):
    """
    Process a CSV or Excel file in chunks to generate column descriptions and save to output CSV.
    
    Args:
        input_file (str): Path to the input file (.csv, .xlsx, .xls).
        chunk_size (int): Number of rows per chunk.
    """
    output_csv = 'app/mapping/output.csv'
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    ext = os.path.splitext(input_file)[1].lower()
    
    # Dictionary to store unique values for each column
    unique_values = {}

    # Read file in chunks
    if ext == '.csv':
        chunks = pd.read_csv(input_file, dtype=str, low_memory=False, chunksize=chunk_size)
    elif ext in ['.xlsx', '.xls']:
        chunks = pd.read_excel(input_file, dtype=str, chunksize=chunk_size)
    else:
        raise ValueError("Unsupported file format.")

    # Process chunks to collect unique values
    for i, chunk in enumerate(chunks):
        chunk.columns = chunk.columns.str.strip()

        # Initialize unique_values for new columns
        if i == 0:
            for col in chunk.columns:
                unique_values[col] = set()

        # Collect unique values for each column
        for col in chunk.columns:
            unique_vals = chunk[col].dropna().unique()
            unique_values[col].update(unique_vals)

        # Free memory
        del chunk
        gc.collect()

    # Generate descriptions for each column
    tasks = []
    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers as needed
        for col in unique_values:
            # Sample up to 30 unique values (or all if fewer)
            sample_vals = list(unique_values[col])[:30] if len(unique_values[col]) > 20 else list(unique_values[col])
            tasks.append(executor.submit(describe_column, col, sample_vals))

        mapping_data = [future.result() for future in as_completed(tasks)]

    # Save results to CSV
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv(output_csv, index=False)
    return output_csv

