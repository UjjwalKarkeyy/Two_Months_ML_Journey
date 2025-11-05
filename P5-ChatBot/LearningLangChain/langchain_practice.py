from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# load api key from .env files
load_dotenv()

# initialize the gemini model
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    api_key = os.getenv('GEMINI_API_KEY'),
)

# create a simple chat prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the following question: {question}"
)

def Chatbot(user_input: str):
    messages = prompt.format_messages(question = user_input)
    response = llm.invoke(messages)
    return response

if __name__ == "__main__":
    print("Gemini Chatbot(type 'exit' to quit)\n")
    while True:
        print("Human Message\n")
        user_input = input()
        # any takes boolean values, returns true if any val is true, else returns false
        if any( word in user_input.lower() for word in ["exit", "quit"]):
            print("Bye Bye!\n")
            break

        chatbot_response = Chatbot(user_input)
        chatbot_response.pretty_print()
