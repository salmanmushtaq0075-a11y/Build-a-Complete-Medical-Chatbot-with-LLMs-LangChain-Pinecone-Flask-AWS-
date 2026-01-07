from langchain.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document



#Extract text from PDFs
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


#filter to minimal docs this will take the entire document and return the source and page content only
from typing import List
from langchain.schema import Document

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Given a list of document objects, return a list of documents objects.containing only 'source' in metadata and the original page content."""
    minimal_docs : list[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
            page_content=doc.page_content,
            metadata={"source": src}
        )
        )
    return minimal_docs 
                        
#Split the documents into smaller chunks for better processing
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunk = text_splitter.split_documents(extracted_data)
    return text_chunk


from langchain.embeddings import HuggingFaceEmbeddings
import torch

def download_embeddings():
    """Download and return HuggingFace embeddings model."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name)
    return embeddings

