import os
import yaml
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from youtubesearchpython import VideosSearch
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup

# Load configuration
def load_config(file_path='config.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

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

# YouTube search and scrape function
def youtube_search_and_scrape(query, max_results=3):
    search = VideosSearch(query, limit=max_results)
    results = search.result()['result']
    scraped_results = []
    for result in results:
        video_url = f"https://www.youtube.com/watch?v={result['id']}"
        try:
            response = requests.get(video_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            description = soup.find('meta', {'name': 'description'})['content']
            # Fetch the transcript
            transcript = YouTubeTranscriptApi.get_transcript(result['id'])
            transcript_text = " ".join([entry['text'] for entry in transcript])
            # Summarize the transcript if necessary (using Ollama)
            if len(transcript_text) > 2000:  # Arbitrary length limit for summarization
                summary_prompt = f"Summarize the following transcript:\n\n{transcript_text}"
                transcript_text = ollama.invoke(summary_prompt)
            scraped_results.append({
                'title': result['title'],
                'url': video_url,
                'description': description,
                'transcript': transcript_text
            })
        except Exception as e:
            print(f"Error scraping {video_url}: {str(e)}")
    return scraped_results

# Vector database search function
def vector_db_search(query, k=3):
    results = vectordb.similarity_search(query, k=k)
    return [{"source": doc.metadata.get('source', 'Unknown'), "content": doc.page_content} for doc in results]

# Combined search function
def combined_search(tool_input):
    if isinstance(tool_input, str):
        try:
            tool_input = eval(tool_input)
        except:
            query = tool_input.strip()
    else:
        query = tool_input.get('query', '')
    if not query:
        return "Error: No query provided."

    # Perform YouTube search and scrape
    youtube_results = youtube_search_and_scrape(query)
    youtube_context = "\n".join([f"Video Title: {result['title']}\nDescription: {result['description']}\nTranscript: {result['transcript']}\nURL: {result['url']}" for result in youtube_results]) if youtube_results else "No relevant YouTube videos found."

    # Perform vector database search
    db_results = vector_db_search(query)
    db_context = "\n".join([f"Source: {result['source']}\nContent: {result['content']}" for result in db_results])

    combined_context = f"YouTube Results:\n{youtube_context}\n\nDatabase Results:\n{db_context}"

    prompt = f"Context: {combined_context}\n\nQuestion: {query}\n\nProvide a comprehensive answer based on both the YouTube and database context. Cite your sources, including video titles and URLs when referencing YouTube content."

    try:
        response = ollama.invoke(prompt)
        return str(response)
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create a combined search tool
combined_search_tool = Tool(
    name="combined_search",
    func=combined_search,
    description="Search YouTube and the vector database for relevant information and provide combined results."
)

# Create agents and tasks from the YAML configuration
agents = []
tasks = []

for agent_config in config['agents']:
    agent = Agent(
        role=agent_config['role'],
        goal=agent_config['goal'],
        backstory=agent_config['backstory'],
        tools=[combined_search_tool],
        verbose=agent_config['verbose'],
        allow_delegation=agent_config['allow_delegation'],
        llm=ollama
    )
    agents.append(agent)

    for task_config in agent_config['tasks']:
        task = Task(
            description=task_config['description'],
            agent=agent,
            expected_output=task_config['expected_output']
        )
        if 'output_file' in task_config:
            task.output_file = task_config['output_file']
        tasks.append(task)

# Create the crew
crew = Crew(
    agents=agents,
    tasks=tasks,
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
