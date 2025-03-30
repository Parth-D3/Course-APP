from pinecone import Pinecone
import ollama
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
import time


load_dotenv()

#### pinecone setup
PINECONE_API_KEY = os.getenv('pinecone')
index_name = "YOUR_INDEX_NAME"

pc = Pinecone(api_key="YOUR_API_KEY")


index = pc.Index(index_name)

#### embedding initialization
embeddings = OllamaEmbeddings(model='nomic-embed-text',  base_url="http://127.0.0.1:11434")

def retrieve_context(query):
    query_embedding = embeddings.embed_query(query)
    results = index.query(vector=query_embedding, top_k=1, include_metadata=True)
    # if results != {}:
    #     return results['matches'][0]['metadata']['text']
    # else:
    #     return "No records found !!!"
    
    return results.get('matches', [{}])[0].get('metadata', {}).get('text', "No records found !!!")
    
def generate_response(query):
    context = retrieve_context(query)

    prompt = f"""
    Restructure the following information to provide a clearer, well-formatted response to the query. 
    Do not include the Query and Retrieved Context in your response. 
    
    Query: {query}
    
    Retrieved Context:
    {context}
    
    Provide a concise and well-structured response.
    
    """

    response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']




while(True):
    query = input("Enter your query: ")
    response = generate_response(query)
    print("\nRESPONSE\n")
    print(response)
    num = int(input("Press 5 to exit: "))
    if num == 5: exit()
