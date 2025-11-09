import glob
import re
from pathlib import Path
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader

# function to load document
def file_reader(path: str, extensions: tuple):
    # give data path
    data_path = Path(path)

    # store all files (list of list)
    files = {}
    # match multiple patterns (e.x: txt, pdf) from subdirectories
    for ext in extensions:
        rglobed_files = (data_path.rglob(f"*.{ext.lower()}")) 
        files[ext] = [file.as_posix() for file in rglobed_files] # as_posix: convert path to standard Unix-style string

    # store all documents (unloaded)
    documents = []
    # initialize a default document loader
    document_loader = None
    for ext, file in files.items():
        if ext == "docx":
            document_loader = [UnstructuredWordDocumentLoader(ele) for ele in file]
        elif ext == "pdf":
            document_loader = [PyPDFLoader(ele) for ele in file]

        documents.extend(loader.load() for loader in document_loader)

    return documents

# function to fetch details from appointment
def appointment_reader(msg: str):
    # Use case-insensitive regex and allow flexible spacing/newlines
    patterns = {
        "name": r"(?i)name\s*[:\-]\s*([A-Za-z ]+)",
        "phone": r"(?i)phone(?: number)?\s*[:\-]\s*(\d{7,15})",
        "email": r"(?i)email(?: address)?\s*[:\-]\s*([\w\.-]+@[\w\.-]+\.\w+)",
        "date": r"(?i)date\s*[:\-]\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"
    }

    extracted = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, msg, re.DOTALL)  # DOTALL makes it work across lines
        extracted[key] = match.group(1).strip() if match else None

    return extracted

'''
                        My Questions
-----------------------------------------------------------------------------
1. What is glob?
Ans: The glob module is a part of Python's standard library and is used to 
find all the pathnames matching a specified pattern. 

2. What is POSIXPath?
Ans: POSIX stands for Portable Operating System Interface — its a standard that defines how operating systems (especially Unix-like ones, like Linux and macOS) handle things like:
        File paths
        Command-line tools
        Permissions
        System calls
    Basically, its a universal rulebook that ensures different Unix-based systems behave consistently. 
3. Difference between extend and append.
Ans:
    Use append() when you want to add a single item (which could be an entire object like another list) as one unit to the end of your list.
    Use extend() when you want to add all the individual items from an iterable (like merging another list) into your existing list.
'''