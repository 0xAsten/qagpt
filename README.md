# QAGPT: Question Answering over Documents with GPT

QAGPT is a question-answering application that uses a combination of document indexing and a large language model (LLM) like GPT to provide accurate and relevant answers to user questions. The system indexes a collection of documents in a vector database and leverages GPT to extract precise answers from the most relevant documents.

## Features

- Indexing of documents in a vector database
- Query engine to find the most relevant documents
- Integration with GPT for answer extraction
- User-friendly interface for inputting questions and viewing answers

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/QAGPT.git
```

2. Navigate to the project directory:

```
cd QAGPT
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

1. Index your documents:

```
python index_documents.py --input_dir /path/to/your/documents --file_type markdown --collection_name my_collection
```

2. Run the application:

```
python app.py
```

3. Open a web browser and navigate to http://localhost:5000 to access the user interface.

4. Enter your question in the search bar and hit Enter or click the "Ask" button.

5. The answer will be displayed along with the most relevant document(s).

## Configuration

- To change the LLM, edit the config.py file and update the LLM_MODEL variable.
- To adjust the number of top documents returned by the query engine, update the NUM_TOP_DOCS variable in the config.py file

## Contributing

We welcome contributions! Please read our CONTRIBUTING.md file to learn how you can contribute to the project.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- OpenAI for providing the GPT models
- Vector database libraries (e.g., Faiss, Annoy, Elasticsearch) for document indexing
- Pre-trained word embeddings (e.g., Word2Vec, GloVe, FastText)

_Note: Be sure to customize the repository URL, username, and other details in the README to match your project._
