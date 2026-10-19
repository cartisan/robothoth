- Three main options for using open weights models: Ollama, vLLM, and SGLang.
- Each engine handles requests differently
## Ollama
- A local user calls the OpenAI-compatible API, and requests line up in a FIFO queue.
-  Ollama runs a pre-quantized GGUF model, a compressed format it pulls, and the response comes back to the user.
- Best for local dev, prototyping, and laptop-scale hardware.
## vLLM
- Many users hit the server at once, and continuous batching slots new requests into the running batch instead of making them wait for it to finish. 
- PagedAttention stores the KV cache, the memory a model keeps for tokens it has already processed.
- Best for high-traffic serving, max GPU utilization, and thousands of concurrent requests.
## SGLang
- Agents and multi-turn chats send requests whose prompts overlap heavily. 
- A prefix-aware scheduler routes them through the RadixAttention cache, a radix tree that reuses every shared prefix instead of recomputing it.
- Built ontop of vLLM
- Best for AI agents and tool loops, multi-turn chats, and JSON/regex outputs.

![[ollama-vllm-sglang-infographic.png]]
##
Associations:
Tags: #📥️ 
Sources: LinkedIn Post
