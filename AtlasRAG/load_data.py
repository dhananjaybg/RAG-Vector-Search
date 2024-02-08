from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAI
import params
from langchain_community.document_loaders import PyPDFLoader



# Set the MongoDB URI, DB, Collection Names

client = MongoClient(params.mongodb_conn_string)
collection = client[params.db_name][params.collection_name]

# Initialize the DirectoryLoader
#loader = DirectoryLoader( './sample_files', glob="./*.txt", show_progress=True)
#data = loader.load()


loader = PyPDFLoader("./sample_pdf/Travelers-2022-Annual-Report-pdf.pdf")
data = loader.load_and_split()


# Define the OpenAI Embedding Model we want to use for the source data
# The embedding model is different from the language generation model
embeddings = OpenAIEmbeddings(openai_api_key=params.openai_api_key)

# Initialize the VectorStore, and
# vectorise the text from the documents using the specified embedding model, and insert them into the specified MongoDB collection
vectorStore = MongoDBAtlasVectorSearch.from_documents( data, embeddings, collection=collection )