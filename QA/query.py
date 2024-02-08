import argparse
import params
import json

from pymongo import MongoClient

from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
#from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import warnings
# Filter out the UserWarning from langchain
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.chains.llm")

# Process arguments
parser = argparse.ArgumentParser(description='Atlas Vector Search Demo')
parser.add_argument('-q', '--question', help="The question to ask")
args = parser.parse_args()

if args.question is None:
    # Some questions to try...
    #query = "How big is the telecom company?"
    #query = "Who started AT&T?"
    #query = "Where is AT&T based?"
    #query = "What venues are AT&T branded?"
    #query = "How big is BofA?"
    #query = "what is BoFa share of all American deposits?"
    #query = "When was the financial institution started?"
    #query = "Does the bank have an investment arm?"
    #query = "Where does the bank's revenue come from?"
    #query = "Tell me about charity."
    #query = "What buildings are BofA branded?"
    #query = "what is salary of the president"
    query = "summarize BoFa journey since inception in 8 bullet points"

else:
    query = args.question

def main():
    
    mongo_client = create_mongo_client(params.mongodb_conn_string)
    collection = mongo_client.get_database(params.db_name).get_collection(params.collection_name)
    vs_ =  CreateVectorStore(collection)
    VectorSearch(query, vs_ ,1 )
    GenerateReponse(query, vs_ , 3 )
    
def CreateVectorStore(collection: object):
    # initialize vector store
    vectorStore = MongoDBAtlasVectorSearch(
        collection, OpenAIEmbeddings(openai_api_key=params.openai_api_key), index_name=params.index_name
    )
    return vectorStore

def VectorSearch(query: str , vectorStore: object , doc_count: int):
    print("Your question:")
    print("-------------")
    print(query)

    
    # perform a similarity search between the embedding of the query and the embeddings of the documents
    print(f"Query Response:") 
    docs = vectorStore.similarity_search_with_score(query, k=doc_count)
    index = 0
    for index, result in enumerate(docs, start = 1):
       print(f"Document : {index} : ") 
       print({result[0].page_content})
    
    return vectorStore

def GenerateReponse(query:str ,vectorStore: object,doc_count: int):
    print("\nAI Response:")
    print("--- ------------ --------")
    
    qa_retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={"k":doc_count},
    )
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = OpenAI(openai_api_key=params.openai_api_key, temperature=0)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=qa_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    x_query = json.dumps({'query':f"{query}"})
    docs = qa(x_query)
    print(docs["result"])
    #print(docs["source_documents"])

def create_mongo_client(uri: str):
    client = MongoClient(uri)
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except Exception as e:
        print(e) 

if __name__ == "__main__":
    main()