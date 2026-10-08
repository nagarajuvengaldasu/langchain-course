import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

from embeddings import PaddedPineconeEmbeddings

load_dotenv()

SOURCE_FILE = "mediumblog1.txt"

if __name__ == "__main__":
    print("Ingesting...")
    loader = UnstructuredLoader(
        file_path=SOURCE_FILE, chunking_strategy="basic", max_characters=1000000
    )
    document = loader.load()

    print("splitting...")
    # Recursive splitter falls back to sentences/words, so chunks never exceed chunk_size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = PaddedPineconeEmbeddings()

    # Fixed IDs per chunk: re-running ingestion overwrites the same vectors
    # instead of adding duplicate copies to the index.
    ids = [f"{SOURCE_FILE}-chunk-{i}" for i in range(len(texts))]

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, ids=ids, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
