
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
import pinecone
import pandas as pd
import csv
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import node_parser
from llama_index.core import Document
from pathlib import Path
from llama_index.readers.file import PandasCSVReader
from ..pinecone.injest_docs import connect_pinecone
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import get_response_synthesizer
from llama_index.core import StorageContext, load_index_from_storage
import json
from ..pinecone.helper import extract_text_from_json
from ..pinecone.injest_docs import inject_docs
import uuid
from ..models import Excel
import pandas as pd
import psycopg2
import re
from psycopg2 import extensions
from ..map_embbeding import upload_to_pinecone
class CustomPandasCSVReader(PandasCSVReader):
    def load_data(self, file: Path, extra_info: dict = None, fs=None) -> list[Document]:
        documents = []
        # Read CSV in chunks
        if fs:
            with fs.open(file) as f:
                chunk_iterator = pd.read_csv(f, chunksize=50, **self._pandas_config)
        else:
            chunk_iterator = pd.read_csv(file, chunksize=50, **self._pandas_config)
        # Process each chunk
        for chunk in chunk_iterator:
            chunk = chunk.fillna("")  # Handle missing values
            # Build a list that starts with the header row
            if not documents:  # Only add headers for the first chunk
                text_list = [" ".join(chunk.columns.astype(str))]
            else:
                text_list = []
            # Append each row’s values
            text_list += chunk.astype(str).apply(lambda row: " ".join(row.values), axis=1).tolist()
            if self._concat_rows:
                documents.append(Document(text="\n".join(text_list), metadata=extra_info or ()))
            else:
                for text in text_list:
                    documents.append(Document(text=text, metadata=extra_info or {}))
 
            break
        return documents
 
 
def upload_to_llama(survey, index='surveys'):
    load_dotenv()
    csv_reader = CustomPandasCSVReader()
    all_text = []
    reader = SimpleDirectoryReader(
            input_files=[survey.file_path],
            file_extractor={".csv": csv_reader}
        )
    
    docs = reader.load_data()
 
  
    pinecone_index = connect_pinecone("test1")
 
    # Ingestion pipeline to extract data and create vector store
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
 
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        documents=docs,
    )
 
    nodes = pipeline.run()
 
    return query_llama(nodes,'test1', query = 'what is the sales of bathingsoap?')
 
def query_llama(vector_store_index, query=''):
    try:
        # Create the query engine from the existing index
        query_engine = vector_store_index.as_query_engine(similarity_top_k=2)
        custom_prompt = f"""
        You are working as a chatbot which gives relevant information/insights.
        1) If you find context irrelevant, then say "Not found relevant context, try again."
        2) If the user query is irrelevant or a greeting, please respond with: "Ask me a question based on the survey/research uploaded."
        3) Generate insights useful for making charts, graphs, documents, presentations, etc.
        4) Just generate insights and information about the data, do not give any other suggestions.
        5) Generate as much numeric data as you can for generating charts.
        6) Give the whole answer in 1 paragraph, without adding extra spaces, and do not use capital letters.
        7) important: generate response of atleast 100 words and all the data like total, percentage etc. (do not write anything else other than information)
        User Query: {query}
        """
        response = query_engine.query(custom_prompt)
       
        return response.response
    except Exception as e:
        print("Error occurred while querying:", e)
        return {"error": str(e)}
 
def handle_csv(survey, index='surveys'):
    convert_file_to_json(survey)
    try:
        # Assuming the survey already exists, update it
        # survey.text = json_data
        survey.status = 'completed'  # Set the status to completed after processing
        survey.save()
    except Exception as e:
        print(f"Survey with id {survey.id} does not exist.")
    
 
 
def convert_file_to_json(survey, chunk_size=10):
    # Variable to store JSON data as a list of dictionaries
    json_data = []
    
    # If file is Excel (.xlsx), handle it differently
    if survey.type == 'xlsx':
        # Open Excel file using openpyxl engine
        xl = pd.ExcelFile(survey.file_path, engine='openpyxl')
        
        # Loop through each sheet (you can adjust to a specific sheet if needed)
        for sheet_name in xl.sheet_names:
            # Read the entire sheet in chunks manually
            # Load the sheet using openpyxl and get the row count to split into chunks
            sheet = xl.parse(sheet_name=sheet_name, header=None)
            total_rows = len(sheet)
            
            chunk_size = 1000  # Define your desired chunk size
            for start_row in range(0, total_rows, chunk_size):
                # End row for the chunk
                end_row = min(start_row + chunk_size, total_rows)
                
                # Get the chunk of rows
                chunk = sheet.iloc[start_row:end_row]
                
                # Convert the chunk to a list of dictionaries and append to json_data
                json_data.extend(chunk.to_dict(orient='records'))
 
 
            # Optionally, you can save the data to a file if needed
            dirname = "json"
            os.makedirs(dirname, exist_ok=True)
            json_file_path = os.path.join(dirname, sheet_name)
            with open(json_file_path, 'w') as json_file:
                json.dump(json_data, json_file)
 
            survey_sheet = Excel.objects.create(
                                name = sheet_name,
                                survey=survey,
                                json_path = json_file_path
                              )
 
            text = extract_text_from_json(json_file_path)
 
            doc_data = {
            "id": str(uuid.uuid4()),
            "survey_id": int(survey.id),
            "survey_name": survey.name,
            "sheet_name": sheet_name,
            "file_type": survey.type,
            "description": survey.description,
            "organization_id": survey.organization_id,
            "uploaded_by": survey.uploaded_by.email,
            }
 
            inject_docs(doc_data=doc_data, extracted_text=text, organisation='surveys')
    
    else:  
 
        for chunk in pd.read_csv(survey.file_path, chunksize=chunk_size):
            # Convert each chunk to a list of dictionaries and append to json_data
            json_data.extend(chunk.to_dict(orient='records'))
 
            break
 
        # Optionally, you can save the data to a file if needed
        dirname = "json"
        os.makedirs(dirname, exist_ok=True)
        json_file_path = os.path.join(dirname, survey.name)
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file)
 
        survey.json_path = json_file_path
        survey.save()
 
        text = extract_text_from_json(json_file_path)
 
        doc_data = {
            "id": str(uuid.uuid4()),
            "survey_id": int(survey.id),
            "survey_name": survey.name,
            "file_type": survey.type,
            "description": survey.description,
            "organization_id": survey.organization_id,
            "uploaded_by": survey.uploaded_by.email,
        }
 
        inject_docs(doc_data, extracted_text = text, organisation='surveys')
 
 
def extract_text_from_csv(file_path):
    text = ''
    row_count = 0  # Counter for the rows
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                text += ', '.join(row) + '\n'
                row_count += 1
                if row_count >= 10:  # Stop after 10 rows
                    break
    except UnicodeDecodeError:
        # Fallback to 'ISO-8859-1' if UTF-8 fails
        with open(file_path, mode='r', encoding='ISO-8859-1') as file:
            reader = csv.reader(file)
            for row in reader:
                text += ', '.join(row) + '\n'
                row_count += 1
                if row_count >= 10:  # Stop after 10 rows
                    break
    return text.strip()
 
 
def insert_csv_in_pinecone(survey):
    if survey.mapping_file_path:
        # If the mapping file path is provided, use it
        upload_to_pinecone(survey.mapping_table_name,"Description",survey)
        
    else:
        text = extract_text_from_csv(survey.file_path)
        with open("extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
   
    
        doc_data = {
            "id": str(uuid.uuid4()),
            "survey_id": int(survey.id),
            "survey_name": survey.name,
            "survey_table_name": survey.table_name,  
            "file_type": survey.type,
            "description": survey.description,
            "organization_id": survey.organization_id,
            "uploaded_by": survey.uploaded_by.email,
        }
        
        inject_docs(extracted_text=text, doc_data=doc_data)