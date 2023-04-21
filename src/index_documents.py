# Langchain wraps the Milvus client and provides a few convenience methods for working with documents. 
# It can split documents into chunks, embed them, and store them in Milvus.

import os
import argparse

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Milvus

def load_documents(file_path, encoding='utf8'):
    loader = TextLoader(file_path, encoding=encoding)
    return loader.load()

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


def index_documents(docs, embeddings, connection_args):
    vector_db = Milvus.from_documents(
        docs,
        embeddings,
        connection_args=connection_args,
    )
    return vector_db


def main(input_dir, encoding, chunk_size, chunk_overlap, host, port):
    # Iterate through all the files in the input directory and process each one
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            print(f"Processing {file_path}...")
            documents = load_documents(file_path, encoding)
            docs = split_documents(documents, chunk_size, chunk_overlap)
            embeddings = OpenAIEmbeddings()
            vector_db = index_documents(docs, embeddings, {"host": host, "port": port})
            print(f"Indexed {len(docs)} chunks from {file_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index documents for Question Answering over Documents application.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing documents to be indexed.')
    parser.add_argument('--encoding', type=str, default='utf8', help='Encoding of the input documents.')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Size of the chunks to split documents into.')
    parser.add_argument('--chunk_overlap', type=int, default=0, help='Number of overlapping characters between consecutive chunks.')
    parser.add_argument('--host', type=str, default="127.0.0.1", help='Host address for the Milvus server.')
    parser.add_argument('--port', type=str, default="19530", help='Port for the Milvus server.')

    args = parser.parse_args()

    main(args.input_dir, args.encoding, args.chunk_size, args.chunk_overlap, args.host, args.port)