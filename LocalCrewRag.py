import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings()

# Create or load the vector database
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def process_documents(directory):
    os.makedirs(directory, exist_ok=True)
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(file_path)
        elif filename.endswith(".pdf"):
            loader = UnstructuredPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue
        documents.extend(loader.load())

    if not documents:
        print("⭐No documents found in the directory.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    vectordb.add_documents(splits)
    vectordb.persist()

# Initialize Ollama with a callback manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
ollama = Ollama(model="llama3", callback_manager=callback_manager)

# Create a RAG tool
def rag_search(**kwargs):
    tool_input = kwargs.get('tool_input', '')
    query = tool_input
    results = vectordb.similarity_search(query, k=3)
    context = "\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in results])
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nProvide an answer based on the context and cite the sources used."
    
    try:
        response = ollama.invoke(prompt)
        return str(response)
    except Exception as e:
        return f"An error occurred: {str(e)}"

rag_tool = Tool(
    name="rag search",
    func=rag_search,
    description="""
    Search the vector database for relevant information and provide cited answers.
    Always use the following schema when calling this tool:
    {
        "tool_name": "rag search",
        "arguments": {
            "tool_input": "your query here"
        }
    }
    """
)

# Create agents with the RAG tool and Ollama LLM
agent1 = Agent(
    role='Beverage Chemist',
    goal='Write a stepwise overview of the best coffee brew process',
    backstory='Expert in beverage chemistry.',
    tools=[rag_tool],
    verbose=True,
    allow_delegation=False,
    llm=ollama,
)

agent2 = Agent(
    role='Professional Brewer and Writer',
    goal='Streamline the coffee brewing process and write a report',
    backstory='Professional brewer with extensive experience.',
    tools=[rag_tool],
    verbose=True,
    allow_delegation=False,
    llm=ollama
)

# Create tasks with explicit instructions on tool usage
task1 = Task(
    description='''
    Write a stepwise overview of the best coffee brew process.
    IMPORTANT: Use the RAG search tool with the following schema:
    {
        "tool_name": "rag search",
        "arguments": {
            "tool_input": "your query here"
        }
    }
    ''',
    agent=agent1,
    expected_output='a detailed guide.'
)

task2 = Task(
    description='''
    Streamline the coffee brewing process and write a report.
    Always print your full answer in proper format. MINIMUM OF 4 Paragraphs.
    IMPORTANT: Use the RAG search tool with the following schema:
    {
        "tool_name": "rag search",
        "arguments": {
            "tool_input": "your query here"
        }
    }
    ''',
    agent=agent2,
    expected_output='A stepwise brewing instruction manual.',
    output_file='The_Perfect_Cup.txt'
)

# Create the crew
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    process=Process.sequential,
    verbose=2
)

if __name__ == "__main__":
    print("⭐Processing documents and updating the vector database...")
    process_documents("./document_directory")
    
    if vectordb._collection.count() == 0:
        print("⭐The vector database is empty. Please add some documents to the 'document_directory' folder and run the script again.")
    else:
        print("⭐Starting the CrewAI workflow...")
        result = crew.kickoff()
        
        print("\nFinal Result:")
        print(result)
