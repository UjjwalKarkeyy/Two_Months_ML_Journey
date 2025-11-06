from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os
import shutil # Important shutil module for high-level file operations

'''
                                        Import Document and Creat Chunks
---------------------------------------------------------------------------------------------------------
'''
# path for new data
DATA_PATH = "./data.txt"

def load_documents():
    # initilize text loader with directory
    document_loader = TextLoader(DATA_PATH)
    # load document and return as a list of document objects
    return document_loader.load()

def split_text(documents: list[Document]):
    """
        Split the text content of the given list of Document objects into smaller chunks.
        Args:
        documents (list[Document]): List of Document objects containing text content to split.
        Returns:
        list[Document]: List of Document objects representing the split text chunks.
    """
    # initialize text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300, # size of each chunk in characters
        chunk_overlap = 100, # overlap between consecutive chunks
        length_function = len, # function to compute the lenght of text
        add_start_index = True, # flag to add start index to each chunk
    )

    # split documents into smaller chunks using text splitter
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # print example of page content and metadata for a chunk
    document = chunks[0]
    print(document.page_content)
    print(document.metadata)
    return chunks # return list of split text chunks


'''
                        Create Chroma DB (Vector DB) and Store Using HugginFace Embedding
-----------------------------------------------------------------------------------------------------------------
'''

# path to the directory to save chroma database
CHROMA_PATH = "./chroma"
def save_to_chroma(chunks: list[Document]):
    """
        Save the given list of Document objects to a Chroma database.
        Args:
        chunks (list[Document]): List of Document objects representing text chunks to save.
        Returns:
        None
    """
    # clear out existing db dir if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # create a new chroma db from the documents using HugginFace embeddings
    db = Chroma.from_documents(
        chunks,
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
        persist_directory = CHROMA_PATH,
    )

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

'''
                        Create Function to Generate & Store Data
-----------------------------------------------------------------------------------------------------------------
'''

def generate_data_store():
    """
        Function to generate vector database in chroma from documents.
    """
    documents = load_documents() # load documents from source
    # # inspect document
    # print(documents[0])
    chunks = split_text(documents) # split documents into manageable chunks
    save_to_chroma(chunks) # save the processed data to chroma db

'''
                        Create Quering Function
-----------------------------------------------------------------------------------------------------------------
'''

def query_rag(query_text):
    """
        Query a Retrieval-Augmented Generation (RAG) system using Chroma database and Gemini.
        Args:
        - query_text (str): The text to query the RAG system with.
        Returns:
        - formatted_response (str): Formatted response including the generated text and sources.
        - response_text (str): The generated response text.
    """
    # use HugginFace Embedding function
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    # prepare DB
    db = Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function = embedding_function,
    )

    # retrive context from the DB using similarity search
    results = db.similarity_search_with_relevance_scores(query_text, k = 3)

    # chek if there are any matching results or if the relevance score is too low
    if len(results) == 0 or results[0][1] < 0.5: # arbirary relevance score threshold
        print(f"Unable to find matching results.")

    # combine context from matching documents
    context_text = "\n\n - - \n\n".join([doc.page_content for doc, _score in results])

    # create prompt template using context and query text
    PROMPT_TEMPLATE = """
        Answer the question based on the following context: {context}
        - - 
        Answer the question based on the above context: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text, question = query_text)

    # initialize Gemini chat model
    model = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        google_api_key = os.environ["GEMINI_API_KEY"],
    )

    # generate response text based on the prompt
    response_text = model.invoke(prompt)

    # get sources of the matching documents
    sources = [doc.metadata.get("source", None) for doc, _score in results]

    # format and return response including generated text and sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, response_text

'''
                        Function Calls & Program Execution
-----------------------------------------------------------------------------------------------------------------
'''
load_dotenv()
# generate data store
generate_data_store()

# query text
query_text = "What is the name of my pet?"

# call query function
formatted_response, response_text = query_rag(query_text)
print(f"The name of your pet is: {response_text.content}")






"""
My questions:

1. What does 'Document' object store?
Ans:

    Document(
        page_content="The actual text content of this chunk.",
        metadata={"source": "filename.pdf", "page": 2}
    )

2. What does k = 3 mean in similarity search?
Ans: 
    When you query your vector database (like Chroma), it finds which stored document embeddings are closest to your query 
    embedding — that is, most semantically similar. Those matches are ranked by similarity score (or “relevance”), and 
    k decides how many of those top matches you want back.
"""