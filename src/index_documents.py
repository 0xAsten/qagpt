# Langchain wraps the Milvus client and provides a few convenience methods for working with documents. 
# It can split documents into chunks, embed them, and store them in Milvus.

import os
import argparse

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.document_loaders import UnstructuredMarkdownLoader

from pymilvus import FieldSchema, DataType, CollectionSchema, Collection, connections, utility


text_field = "otext"
primary_field = "id"
vector_field = "embedding"


def load_documents(file_path, encoding='utf8', file_type='text'):
    if file_type == 'markdown':
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding=encoding)
    return loader.load()


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


def index_documents(milvus, docs):
    # Index the documents using the provided Milvus instance
    milvus.add_documents(docs)

def create_milvus_collection(embeddings, collection_name, host, port):
    # Connect to Milvus instance
    if not connections.has_connection("default"):
        connections.connect(host=host, port=port)
    utility.drop_collection(collection_name)
    # Create the collection in Milvus
    fields = []
    # Create the metadata field
    fields.append(
        FieldSchema('source', DataType.VARCHAR, max_length=200)
    )
    # Create the text field
    fields.append(
        FieldSchema(text_field, DataType.VARCHAR, max_length=1500)
    )
    # Create the primary key field
    fields.append(
        FieldSchema(primary_field, DataType.INT64, is_primary=True, auto_id=True)
    )
    # Create the vector field
    fields.append(FieldSchema(vector_field, DataType.FLOAT_VECTOR, dim=1536))
    # Create the schema for the collection
    schema = CollectionSchema(fields)
    # Create the collection
    collection = Collection(collection_name, schema)
    # Index parameters for the collection
    index = {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"M": 8, "efConstruction": 64},
    }
    # Create the index
    collection.create_index(vector_field, index)

    # Create the VectorStore
    milvus = Milvus(
        embeddings,
        {"host": host, "port": port},
        collection_name,
        text_field,
    )

    return milvus

def main(input_dir, encoding, chunk_size, chunk_overlap, host, port, file_type, collection_name):
    embeddings = OpenAIEmbeddings()
    milvus = create_milvus_collection(embeddings, collection_name, host, port)
    # Iterate through all the files in the input directory and process each one
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            print(f"Processing {file_path}...")
            documents = load_documents(file_path, encoding, file_type)
            docs = split_documents(documents, chunk_size, chunk_overlap)
            index_documents(milvus, docs)
            print(f"Indexed {len(docs)} chunks from {file_path}.")


# python index_documents.py --input_dir /path/to/your/documents --file_type markdown --collection_name my_collection
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index documents for Question Answering over Documents application.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing documents to be indexed.')
    parser.add_argument('--encoding', type=str, default='utf8', help='Encoding of the input documents.')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Size of the chunks to split documents into.')
    parser.add_argument('--chunk_overlap', type=int, default=0, help='Number of overlapping characters between consecutive chunks.')
    parser.add_argument('--host', type=str, default="127.0.0.1", help='Host address for the Milvus server.')
    parser.add_argument('--port', type=str, default="19530", help='Port for the Milvus server.')
    parser.add_argument('--file_type', type=str, default="text", choices=["text", "markdown"], help='Type of the input files (text or markdown).')
    parser.add_argument('--collection_name', type=str, required=True, help='Name of the collection to index the documents into.')

    args = parser.parse_args()

    main(args.input_dir, args.encoding, args.chunk_size, args.chunk_overlap, args.host, args.port, args.file_type, args.collection_name)