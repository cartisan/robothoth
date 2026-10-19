## General Overview
- Components
	- LLM Wrappers
		- LLMs: predict a response string given an input string
			- method: `.predict`
		- ChatModels: predict a response message given a list of input messages
			- ChatMessage has `content` and `role`
			- can use classes like `HumanMessage` to wrap around that
	- Prompt Templates
		- Make prompts modular
		- classes like `ChatPromptTemplate`, can take a templated f-string like `f"Hello {name}"` using `.from_template`
		- then can use `.format_messages(name="Tim")` to create an actual message
	- Parsers
		- Use an llm to parse a message into a structured format
		- Use response schema instances to define individual output fields like: `ResponseSchema(name="field_name", description="Command to llm what to extract")`
		- Use `StructuredOutputParser.from_response_schemas(schema_list)` to create a parser, and use `.get_format_instructions` to get an LLM prompt for structured output generation
		- use `.parse` to parse an response into a dict
		- Can also use pydantic classes to define the schema, as long as you set the `description` property and use `PydanticOutputParser`
			- allows to use validators on schemas
	- Indexes
- Memory
	- `ConversationBufferMemory` or `ConversationBufferWindowMemory` - stores messages, potentially last k
		- has to be assigned to a conversation chain
			```python
			llm = ChatOpenAI(model="gpt-5-nano")
			memory = ConversationBufferMemory()
			conversation = ConversationChain(llm=llm, memory=memory)
			conversation.invoke("test, test")
			```
	- `Entity` - can use LLM to extract information on entities used in a conversation
- Chains
	- sequential chains: define state machine of components
	```python
		prompt_hi=ChatPromptTemplate.from_template("Say hi to {name}")
		prompt_translate=ChatPromptTemplate.from_template("translate {text} to Russian.")
		chain = prompt_hi | client | prompt_translate | client | StrOutputParser()
		chain.invoke(input={"name":"Leon"})
	```
	- in case of multiple variables in the second prompt, use `RunnablePassthrough`
	- router chain: conditional routing to specialized chains
- Agents
	- allow LLMs to take actions
##
Associations: [[RAG]], [[python]]
Tags: #📥️ 
Sources:
