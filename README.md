For my brokies who hate API keys and wallet bloodletting,

AI agents can be semantically picky, so make sure you take advantage of the crew AI tools like expected output and make your descriptions stepwise and concise for best results. I like telling them to cite sources, but you don't have to. You should be able to put just about any file in there that we can read, and it will be added to the vector database and support the agents' reasoning and justification in answers and thoughts. I recommend leaving the small guide rails for tool calling and publishing of work in the task descriptions, however, this shouldn't be strictly necessary anymore. I also recommend a large context window LLM wherever possible (currently using "mistral-nemo:latest" 7B, 128k Context Window) , but you should already know that. Implementations of other tools like DuckDuckGo search and website scraping may come in the future.

The LocalCrewRag is the original RAG team with agent info built into the program
The CombinedTool team is an updated and polished framework that operates the same way but with updated dependencies and the ability to scrape YouTube transcripts in conjunction with referencing the VectorDB. It has the added advantage of pointing to the config YAML for task and agent data, making repurposing and modifying the team easier.

Python 3.11.6 in a virtual environment (.venv) was used in my case. However, using conda and other Python versions should not affect the functionality.

Put your research team to Work!
