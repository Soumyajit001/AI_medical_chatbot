from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



data_path = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob = '*pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(data=data_path)
print("Length of the pdf: ", len(documents))



def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print("Length of the chunks: ", len(text_chunks))



# Download the Embeddings from Hugging Face
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

    return embeddings

embeddings = download_hugging_face_embeddings()
# sentence-transformers model: It maps sentences and paragraphs to a 384 dimensional dense vector space
# and can be used for tasks like clustering or semantic search.


query_results = embeddings.embed_query("Hello world!")
print("Length of query: ", len(query_results))

DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embeddings)
db.save_local(DB_FAISS_PATH)