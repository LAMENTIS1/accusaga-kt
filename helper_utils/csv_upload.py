from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PandasCSVReader
from llama_index.core.ingestion import IngestionPipeline
import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import Pinecone, ServerlessSpec
import importlib
from pathlib import Path
from llama_index.readers.file.tabular.base import PandasExcelReader
from ..pinecone.helper import get_file_path, get_cloud_files, remove_downloaded_file



# Load environment variables from a .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# Llamaindex global settings for llm and embeddings
EMBED_DIMENSION = 512
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=EMBED_DIMENSION)

class SingleSheetExcelReader(PandasExcelReader):
    def load_data(self, file: Path, sheet_name: str, extra_info: dict = None):
        # Ensure pandas can handle Excel files
        if not pd:
            raise ImportError("Pandas is required to read Excel files.")
        xls = pd.ExcelFile(file)
        # Directly read the specified sheet from the Excel file
        try:
            df = pd.read_excel(file, sheet_name=sheet_name)
            df = df.fillna("")
        except Exception as e:
            print(f"Error reading sheet '{sheet_name}': {e}")
            return []
        # Construct the document text from DataFrame rows
        header = " ".join(df.columns.astype(str))
        rows = df.astype(str).apply(lambda row: " ".join(row), axis=1).tolist()
        document_text = "\n".join([header] + rows)
        # Create a single Document containing all rows from the sheet
        document = Document(text=document_text, metadata=extra_info or {})
        return [document]
    

class CustomPandasCSVReader(PandasCSVReader):
    def load_data(self, file: Path, extra_info: dict = None, fs=None) -> list[Document]:
        # Read CSV using pandas read_csv
        if fs:
            with fs.open(file) as f:
                df = pd.read_csv(f, **self._pandas_config)
        else:
            df = pd.read_csv(file, **self._pandas_config)
        df = df.fillna("")
        documents = []
        # Build a list that starts with the header row
        text_list = [" ".join(df.columns.astype(str))]
        # Append each row’s values
        text_list += df.astype(str).apply(lambda row: " ".join(row.values), axis=1).tolist()
        if self._concat_rows:
            documents.append(Document(text="\n".join(text_list), metadata=extra_info or ()))
        else:
            for text in text_list:
                documents.append(Document(text=text, metadata=extra_info or {}))
        return documents



def connect_pinecone(index_name = 'llama'):
    # Load environment variables
    load_dotenv()
    index_name = str(index_name)

    # Pinecone API key
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    # Initialize Pinecone client by creating an instance of the Pinecone class
    pc = Pinecone(api_key=pinecone_api_key)
    

    try:
        # Check if the index exists, create it if not
        if index_name not in pc.list_indexes().names():
            pass            
        
        # Connect to the Pinecone index
        index = pc.Index(index_name)
        return index
    except Exception as e:
        print(f"Error connecting to Pinecone index: {e}")
        return None


def extract_and_upload_csv_document(survey, user_query):
    # Read the CSV file
    # file_path = get_cloud_files(survey.cloud_file_path)
    file_path = survey.file_path
    # data = pd.read_csv(file_path)
    file_extension = survey.type

    # Create the Pinecone index if it doesn't exist
    pinecone_index = connect_pinecone()

    if pinecone_index is None:
        print("Failed to connect to Pinecone index. Aborting document injection.")
        return

    if file_extension == 'csv':
        csv_reader = CustomPandasCSVReader()
        reader = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor={".csv": csv_reader}
        )
    else:  
        excel_reader = CustomPandasCSVReader()  
        reader = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor={".xls": excel_reader, ".xlsx": excel_reader}
        )

    docs = reader.load_data()

    # Check a sample chunk
    # Ingestion pipeline to extract data and create vector store
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        documents=docs,
    )
    nodes = pipeline.run()
    return query_llama(nodes, query = user_query)

    
def extract_and_upload_excel_document(sheet_name, survey, user_query):
    # Read the Excel file
    file_path = get_cloud_files(survey.cloud_file_path)
    file_extension = survey.type

    # Create the Pinecone index if it doesn't exist
    pinecone_index = connect_pinecone()
    if pinecone_index is None:
        print("Failed to connect to Pinecone index. Aborting document injection.")
        return

    if file_extension in ['xls', 'xlsx']:

        # Use openpyxl for reading .xlsx or xlrd for .xls files
        reader = SingleSheetExcelReader()
        docs = reader.load_data(Path(file_path), sheet_name=sheet_name)
    else:  
        csv_reader = CustomPandasCSVReader()
        reader = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor={".csv": csv_reader}
        )

        docs = reader.load_data()

    # Ingestion pipeline to extract data and create vector store
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)


    pipeline = IngestionPipeline(
        vector_store=vector_store,
        documents=docs,
    )
    nodes = pipeline.run()
    # remove_downloaded_file(file_path)
    return query_llama(nodes, query=user_query)



def query_llama(nodes,query=''):
    # Create the query engine (this can be done once, cached if needed)


    pinecone_index = connect_pinecone()
    
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)    
    vector_store_index = VectorStoreIndex(nodes)
    
    try:
        query_engine = vector_store_index.as_query_engine(similarity_top_k=2)
        # Run the query on the index and get the response
        custom_prompt = f"""
        You are working as a chatbot which gives relevant information/insights. 
        1) If you find context irrelevant, then say "Not found relevant context, try again."
        2) If the user query is irrelevant or a greeting, please respond with: "Ask me a question based on the survey/research uploaded."
        3) Generate insights useful for making charts, graphs, documents, presentations, etc.
        4) Just generate insights and information about the data, do not give any other suggestions.
        5) Generate as much numeric data as you can for generating charts.
        6) Give the whole answer in 1 paragraph, without adding extra spaces, and do not use capital letters.
        7) important: generate response of atleast 100 words and all the data like total, percentage etc. (do not write anything else other than information)
        8) do not give individual numbers until query is specific.

        User Query: {query}
        """
        
        response = query_engine.query(custom_prompt)
        
        # Return the query response
        return response.response
    except Exception as e:
        print("Error occurred while querying:", e)
        return {"error": str(e)}
