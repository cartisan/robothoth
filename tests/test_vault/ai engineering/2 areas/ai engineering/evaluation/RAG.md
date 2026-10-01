openai API key: OPENAI_API_KEY
```
[redacted API key]
```
## Embeddings
- placing an object into a specialized space
- in LLM applications, we use highly multi dimensional spaces
- we learn embedings for words in such a way, that related concepts are close in the space
	- ideally, also allowing for semantic preserving vector operations
- we embed words, images and so on into such spaces
	- e.g. sentence: turn each token into a vector, compute mean of all vectors
		- called mean pooling
		- reason: normalizes vector sum by sentence length
	- ideally: use sentence transformers, so word order is not lost
## Chunking
- for larger documents, chunk the text
- length depends on:
	- document size
	- embedding model and token limits
	- expected user queries: short & specific vs long & detailed
		- short chunks capture precise meaning but miss context
		- long chunks capture context but might be too generic
	- use case
- e.g. RecursiveCharacterTextSplitter
## Vector Databases
- efficiently store highly multi-dimensional vectors
- use similarity based operations to retrieve vectors similar to a request
	- e.g. cosine similarity
- examples: postgres + pgvector
	- just use three column table: id, content\:text , embedding\:vector(1536)
- Approach:
	- User request -> vectorization -> query relevant content from db -> use as context
	- takes a vector, returns text chunks related to that vector based on their vectors
- Benefits
	- allow to serve your own content, keep it relevant and up to date
	- summarize and safe conversations
	- reduce hallucinations
	- reduce API calls, token usage

##
Associations:
- [[z> MOC AI Evaluation]]
- [[RAG Evaluation]]
- [[default_vault/0 inbox/z> MOC ML]]
Tags: #📥️ 
Sources:
-  [https://www.freecodecamp.org/news/ai-evaluation-engineering-build-a-production-grade-llm-evaluation-platform-handbook](https://www.freecodecamp.org/news/ai-evaluation-engineering-build-a-production-grade-llm-evaluation-platform-handbook/#heading-part-4-rag-evaluation-the-six-metrics-that-carry-all-the-diagnostic-weight)
