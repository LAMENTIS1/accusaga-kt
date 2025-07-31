import os
from dotenv import load_dotenv
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.core import SimpleDirectoryReader
from llama_cloud.client import LlamaCloud
from llama_cloud.types import CloudPineconeVectorStore
from llama_cloud.types import CloudDocumentCreate
import uuid
import os
import zipfile
import pandas as pd
import json
import shutil

load_dotenv()

# Initialize the LlamaCloud client
client = LlamaCloud(
    token=os.getenv("LLAMA_CLOUD_API_KEY")
    )


def llama_cloud_injest(text, docdata, path, file_id, pipeline_id="f772221a-928b-459f-9083-5ca601a0cb08"):
    try:
        with open(path, 'rb') as f:
            file = client.files.upload_file(upload_file=f, project_id="58e8e9d8-9495-439d-8629-7c6b32471e51")

        files = [
        {'file_id': file.id}
        ]

        embedding_config = {
        'type': 'OPENAI_EMBEDDING',
        'component': {
            'api_key': os.getenv("OPENAI_API_KEY"), 
            'model_name': 'text-embedding-ada-002' 
        }
    }

        transform_config = {
        'mode': 'auto',
        'config': {
            'chunk_size': 1024, 
            'chunk_overlap': 20 
        }
    }

        pipeline = {
        'name': 'test-pipeline',
        'embedding_config': embedding_config,
        'transform_config': transform_config,
        'file_info': files, 
        'pipeline_id': '1547f7a9-2475-4292-a423-cbc150cf1be6' 
    }
        
        pipeline_files = client.pipelines.add_files_to_pipeline(pipeline.get("pipeline_id"), request=files)
    except Exception as e:
        print(f"An error occurred: {e}")


def split_csv(file_path, chunk_size=100):  # chunk_size is the number of rows per chunk
    """Split CSV into smaller chunks"""
    chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
    
    chunk_files = []
    for i, chunk in enumerate(chunk_iter):
        chunk_filename = f"{file_path.split('.')[0]}_part{i+1}.csv"
        chunk.to_csv(chunk_filename, index=False)
        chunk_files.append(chunk_filename) 
        break
    return chunk_files


def compress_file(file_path):
    """Compress the CSV file into a zip file"""
    zip_file = file_path + ".zip"
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, arcname=file_path.split('/')[-1])
    return zip_file


def csv_to_json(input_file, output_file):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Create a list to hold the JSON objects
    json_data = []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Create a dictionary for each row, only including non-empty values
        row_data = {col: row[col] for col in df.columns if pd.notna(row[col]) and row[col] != ""}
        
        # Add the dictionary to the list
        json_data.append(row_data)
    
    # Write the JSON data to a file
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=4)
    

def llama_cloud_injest_csv(survey):
    try:
        # Check the size of the file before uploading
        file_size = os.path.getsize(survey.file_path)
        
        # Convert CSV to JSON format
        json_file_path = survey.file_path.replace(".csv", ".json")  # Define the output JSON file path
        
        # Convert the CSV file to JSON
        csv_to_json(survey.file_path, json_file_path)

        # If the file is too large (e.g., > 100MB), split it into smaller chunks (if needed)
        chunk_files = []
        if file_size > 100 * 1024 * 1024:  # 100MB threshold
            chunk_files = split_csv(survey.file_path)  # Assuming you have a split_csv function
        else:
            chunk_files = [json_file_path]  # Single file, no need to split

        uploaded_files = []
        # Upload each chunk (or the original file if it's small enough)
        for chunk_file in chunk_files:
            with open(chunk_file, 'rb') as f:
                file = client.files.upload_file(upload_file=f, project_id="58e8e9d8-9495-439d-8629-7c6b32471e51")

            uploaded_files.append({'file_id': file.id, 'survey_id':survey.id})

        files = uploaded_files  # List of all uploaded files (chunks or original)

        embedding_config = {
            'type': 'OPENAI_EMBEDDING',
            'component': {
                'api_key': os.getenv("OPENAI_API_KEY"), 
                'model_name': 'text-embedding-ada-002' 
            }
        }

        transform_config = {
            'mode': 'auto',
            'config': {
                'chunk_size': 1024, 
                'chunk_overlap': 20 
            }
        }

        pipeline = {
            'name': 'test-pipeline',
            'embedding_config': embedding_config,
            'transform_config': transform_config,
            'file_info': files, 
            'pipeline_id': '1547f7a9-2475-4292-a423-cbc150cf1be6'  
        }

        # Add files to pipeline (processed from uploaded files)
        pipeline_files = client.pipelines.add_files_to_pipeline(pipeline.get("pipeline_id"), request=files)

    except Exception as e:
        print(f"An error occurred: {e}")


import os
def create_pipeline():

    load_dotenv()
    client = LlamaCloud(
    token=os.getenv("LLAMA_CLOUD_API_KEY"),
)
    """Create pipeline."""
        
    ds = {
    'name': 'pinecone',
    'sink_type': 'PINECONE',
    'component': CloudPineconeVectorStore(api_key=os.getenv("PINECONE_API_KEY"), index_name='test-index')
    }
    data_sink = client.data_sinks.create_data_sink(request=ds)
    embedding_config = {
    'type': 'OPENAI_EMBEDDING',
    'component': {
        'api_key': '<YOUR_API_KEY_HERE>', # editable
        'model_name': 'text-embedding-ada-002' # editable
    }
    }

    # Transformation auto config
    transform_config = {
        'mode': 'auto',
        'config': {
            'chunk_size': 1024, # editable
            'chunk_overlap': 20 # editable
        }
    }

    pipeline_req = {
      'name': 'test-pipeline',
      'configured_transformations': embedding_config,
      'data_sink': data_sink
    }
    pipeline = client.pipelines.upsert_pipeline(request=pipeline)
    return pipeline


def llama_cloud_query(query):
    index = LlamaCloudIndex("test-pipeline", project_name="Default")
  
    query_engine = index.as_query_engine()

    custom_prompt = f"""
        You are working as a Insightful person which gives relevant information/insights. 
        note: give answer in percentage, ratio etc..
        User Query: {query}
        """
    
    retriever = index.as_retriever()
    nodes = retriever.retrieve(query)



    file_name = nodes[0].metadata.get('file_name')

    
    answer = query_engine.query(custom_prompt)

    return answer, file_name