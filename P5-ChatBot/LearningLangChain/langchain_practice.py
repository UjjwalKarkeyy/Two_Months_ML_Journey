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

conversation = []
def Chatbot(user_input: str):
    conversation.append(("human", user_input))
    messages = [{"role": "assistant", "content": "You are a helpful assistant."}]
    for role, content in conversation:
        messages.append({"role": role, "content": content})
    response = llm.invoke(messages)
    conversation.append(("assistant", response.content))
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