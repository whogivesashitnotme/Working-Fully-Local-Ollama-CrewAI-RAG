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
        print("⭐ No documents found in the directory.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    vectordb.add_documents(splits)
    vectordb.persist()

# Initialize Ollama with a callback manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
ollama = Ollama(model="llama3", callback_manager=callback_manager)

# Create a RAG tool
def rag_search(tool_input):
    query = tool_input
    if isinstance(tool_input, dict):
        query = tool_input.get('tool_input', '')
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
    description="Search the vector database for relevant information and provide cited answers."
)

# Create agents with the RAG tool and Ollama LLM
agent1 = Agent(
    role='Beverage Chemist',
    goal='Analyze coffee brewing chemistry and process to optimize flavor and quality.',
    backstory='Expert in beverage chemistry.',
    tools=[rag_tool],
    verbose=True,
    allow_delegation=False,
    llm=ollama,
)

agent2 = Agent(
    role='Writer',
    goal='Create a detailed brewing manual for the best coffee',
    backstory='Experienced brewer and writer.',
    tools=[rag_tool],
    verbose=True,
    allow_delegation=False,
    llm=ollama
)

# Create tasks with explicit instructions on tool usage
task1 = Task(
    description='''
    Analyze coffee brewing chemistry and process for drip, espresso, french press, and Turkish brewing methods. Then write an overview for optimal brewing in all use cases. Cite your sources.

    If you have to use the RAG search tool in your work, ask your question directly in normal sentence format: 'your full query here'.
    Once finished, place your full answer after "Final Answer:"
    ''',
    agent=agent1,
    expected_output='Chemical and process report for coffee brewing.'
)

task2 = Task(
    description='''
    Develop a stepwise brewing manual covering all brewing methods and desired brew strengths based on the information received. Cite your sources.

    If you have to use the RAG search tool in your work, ask your question directly in normal sentence format: 'your full query here'.
    Once finished, place your full answer after "Final Answer:"
    ''',
    agent=agent2,
    expected_output='Detailed brewing instruction manual.',
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
    print("⭐ Processing documents and updating the vector database...")
    process_documents("./Feed")
    
    if vectordb._collection.count() == 0:
        print("⭐ The vector database is empty. Please add some documents to the 'Feed' folder and run the script again.")
    else:
        print("⭐ Starting the CrewAI workflow...")
        result = crew.kickoff()
        
        print("\nFinal Result:")
        print(result)
