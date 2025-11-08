from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from utils import file_reader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import MultiQueryRetriever
import os
import shutil
import warnings

"""
                        Ignoring Warnings for Score
-----------------------------------------------------------------------------------------------
"""
warnings.filterwarnings(
    "ignore",
    message="The method `BaseRetriever.get_relevant_documents` was deprecated in langchain-core 0.1.46 and will be removed in 1.0. Use :meth:`~invoke` instead",
    category=UserWarning
)

"""
                        Load API Key from Environment (.env)
-----------------------------------------------------------------------------------------------
"""
# load api key from .env files
load_dotenv()

"""
                                Load Documents (RAG -> Retrieval)
-----------------------------------------------------------------------------------------------
"""
# path of data
data_path = "./data"

# call file reader from utils.py
# Note: Can contain list of lists
loaded_files = file_reader(path = data_path, extensions=("docx", "pdf"))

"""
                                Create Chunks
-----------------------------------------------------------------------------------------------
"""

def split_text(documents: list[Document]):
    # initialize text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 100,
        length_function  = len,
        add_start_index = True,
    )

    # split documents into smaller chunks using text splitter
    chunks = text_splitter.split_documents(documents)
    # print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

"""
                                Save Documents as Vector Embeddings in Chroma DB
-----------------------------------------------------------------------------------------------
"""

# path to the dir to save chroma db
CHROMA_PATH = "./chroma"
def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    for chunk in chunks:
        Chroma.from_documents(
            chunk,
            HuggingFaceEmbeddings(
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            ),
            persist_directory = CHROMA_PATH,
        )
        # print("Chunk saved!")

"""
                Function Calls: Load Documents, Create Chunks, & Save to Chroma DB
-----------------------------------------------------------------------------------------------
"""

def generate_data_store():
    chunks = []
    documents = loaded_files
    for doc in documents:
        chunks.append(split_text(doc))
    save_to_chroma(chunks)

"""
                                RAG -> Augmentation
-----------------------------------------------------------------------------------------------
"""
embedding_function = None
db = None

def load_rag():
    
    # use HuggingFace Embedding Function
    embedding_function = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

    # prepare chroma DB
    return Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function = embedding_function,
    )

"""
            RAG -> Generation + Temporary Chatbot Memory (Using Simple List Append)
-----------------------------------------------------------------------------------------------
"""

conversation = []
def BainiAI(user_input: str):
    # print(user_input)
    # retrive context from the DB using multiquery retriever
    results = multi_retriever.get_relevant_documents(user_input)
    # check if there are results
    if not results:
        print(f"Unable to answer that!")

    else:
        # combine context from matching documents
        context_text = "\n\n - - \n\n".join([doc.page_content for doc in results])
        # print(context_text)

        # create prompt template using context and query text
        PROMPT_TEMPLATE = """
            Answer the question based on the following context: {context}
            - - 
            You are BainiAI, an intelligent virtual assistant for TechNova Solutions Pvt. Ltd.
            Your role is to help employees and customers learn about the company and its policies.
            The information you rely on comes strictly from the provided documents (i.e., context provided above), 
            which include the company overview and departmental policies such as HR and IT: {question}
        """
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context = context_text, question = user_input)
        conversation.append(("human", user_input))
        response = llm.invoke(prompt)
        conversation.append(("assistant", response.content))
        return response

"""
                                Program's Main Execution
-----------------------------------------------------------------------------------------------
"""

if __name__ == "__main__":
    generate_data_store()
    db = load_rag()

    # initialize the gemini model
    llm = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        api_key = os.getenv('GEMINI_API_KEY'),
    )

    # initialize the retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})
    multi_retriever = MultiQueryRetriever.from_llm(
    retriever = retriever,
    llm = llm,
)

    print("bainiAI(type 'exit' to quit)\n")
    while True:
        print("Human Message\n")
        user_input = input()
        # any takes boolean values, returns true if any val is true, else returns false
        if any( word in user_input.lower() for word in ["exit", "quit"]):
            print("Bye Bye!\n")
            break
        chatbot_response = BainiAI(user_input)
        print("Maybe you want to know about our company?") if chatbot_response == None else chatbot_response.pretty_print()




"""
                My Questions
---------------------------------------------------------------------------------
1. What is InMemorySaver?
Ans:
    InMemorySaver is a “checkpointer” implementation in the LangGraph system (part of LangChain). 
    It implements the interface for saving “checkpoints” of graph state, but stores everything in memory (RAM). 
    A “checkpoint” here means the snapshot of a graph’s state at a given moment (for example: conversation history, 
    other stateful data) tied to a thread_id (so that you can resume a thread later). Because it lives only in memory, once 
    your process ends or is restarted, the stored state is lost. This means InMemorySaver is non-persistent.

"""