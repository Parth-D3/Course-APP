import warnings
warnings.filterwarnings("ignore")
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# ENVIRONMENT SETUP
load_dotenv()

PINECONE_API_KEY = os.getenv('pinecone')
OLLAMA_URL = "http://localhost:11434"  

# PINECONE SETUP
pinecone = Pinecone(api_key="YOUR_API_KEY")
index_name = "YOUR_INDEX_NAME"
index = pinecone.Index(index_name)


embeddings = OllamaEmbeddings(model='nomic-embed-text') 

def generate_embeddings(text_chunks):
   
    return embeddings.embed_documents(text_chunks)

def insert_embeddings_to_pinecone(index, text_chunks, metadata_list=None):
    """
    Insert embeddings into Pinecone.
    Args:
        index: Pinecone index object.
        text_chunks: List of text chunks to embed and insert.
        metadata_list: Optional list of metadata dictionaries (one per text chunk).
    """
    vectors = []
    embedding_vectors = generate_embeddings(text_chunks)

    
    for i, embedding in enumerate(embedding_vectors):
        vector_id = f"chunk-{i}"
        metadata = metadata_list[i] if metadata_list else {}

       
        vectors.append({"id": vector_id, "values": embedding, "metadata": metadata})

  
    index.upsert(vectors)

    print("Uploaded")
def process_pdf(file_path):
    """
    Load and split a single PDF into chunks, then insert embeddings into Pinecone.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Prepare text and metadata
    text_chunks = [chunk.page_content for chunk in chunks]
    metadata_list = [{"text":chunk.page_content, "source": file_path, "page": chunk.metadata.get("page", 0)} for chunk in chunks]


    insert_embeddings_to_pinecone(index, text_chunks, metadata_list)

def process_all_pdfs(folder_path):
    """
    Process all PDF files in a folder.
    """
    for filename in os.listdir(folder_path):
        
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file: {file_path}")
        process_pdf(file_path)
    
    print("All PDFs processed successfully.")

if __name__ == "__main__":
    folder_path = "data\\pdfs"
    process_all_pdfs(folder_path)
    print("\n\nSuccess\n\n")
