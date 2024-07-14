For my Brokies who hate API keys and wallet bloodletting

AI agents can be semantically picky, so make sure you take advantage of the crew AI tools like expected output and make your descriptions stepwise and concise for best results. I like telling them to cite sources, but you don't have to. You should be able to put just about any file in there that we can read, and it will be added to the vector database and support the agents' reasoning and justification in answers and thoughts. Don't remove the schema for the RAG tool from the task and agent description, or they might forget how to properly call it and will throw errors or loop. I recommend a large context window LLM wherever possible, but you should already know that. Implementations of other tools like DuckDuckGo search or YouTube video strippers, etc., may come in the future.

Python 3.11.6 in a .venv was used in my case however conda and other python versions should not be an issue for functionality.
