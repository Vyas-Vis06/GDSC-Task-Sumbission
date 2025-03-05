import os
import openai
from google.colab import files
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.utilities import SerpAPIWrapper

# Step 1: Install necessary dependencies (for Google Colab)
!pip install langchain openai faiss-cpu google-search-results pypdf

# Step 2: Set up API Keys (Replace with your own keys)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["SERPAPI_API_KEY"] = "your-serpapi-api-key"

# Step 3: Initialize OpenAI LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

# Step 4: Define FileProcessor class for file ingestion
class FileProcessor:
    def __init__(self):
        self.documents = []
        self.vectorstore = None

    def load_file(self, file_path):
        # Load PDF or text files
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        self.documents.extend(loader.load())
        self.vectorstore = FAISS.from_documents(self.documents, OpenAIEmbeddings())

    def get_retriever(self):
        if self.vectorstore is None:
            raise ValueError("No file loaded. Please load a file first.")
        return self.vectorstore.as_retriever()

# Step 5: Define Basic Arithmetic Functions
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b if b != 0 else "Cannot divide by zero"

# Step 6: Define Summarization Function
def summarize_text(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Summarize the following text:"},
            {"role": "user", "content": text}
        ]
    )
    return response["choices"][0]["message"]["content"]

# Step 7: Define Web Search Function
search = SerpAPIWrapper()

def web_search(query):
    return search.run(query)

# Step 8: Setting up RAG
file_processor = FileProcessor()
retriever = None  # Will be set after loading a file

# Step 9: Define Tools for Function Calling
tools = [
    Tool(name="Addition", func=add, description="Performs addition of two numbers"),
    Tool(name="Subtraction", func=subtract, description="Performs subtraction of two numbers"),
    Tool(name="Multiplication", func=multiply, description="Performs multiplication of two numbers"),
    Tool(name="Division", func=divide, description="Performs division of two numbers"),
    Tool(name="Summarization", func=summarize_text, description="Summarizes a given text"),
    Tool(name="Web Search", func=web_search, description="Searches the web for relevant information")
]

# Step 10: Initialize AI Agent
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Step 11: Define Function to Query Agent
def query_agent(query):
    global retriever
    if retriever is None:
        return "No file loaded. Please load a file first."
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)

# Step 12: Define File Upload Handling for Google Colab
def upload_and_process_file():
    uploaded = files.upload()
    for file_name in uploaded.keys():
        file_processor.load_file(f"/content/{file_name}")
        global retriever
        retriever = file_processor.get_retriever()
        print(f"File {file_name} loaded successfully!")

# Example Usage
# upload_and_process_file()  # Upload file for retrieval in Google Colab
# response = query_agent("Summarize the document.")
# print(response)
# web_result = agent.run("Search the web for latest AI research trends.")
# print(web_result)
