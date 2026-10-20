## Failure Modes
- **Retrieval Failures**: System fails to retrieve all relevant context, or retrieves irrelevant ones
- **Generation Failure**: System fails to produce an answer that is grounded in retrieved context
- **Reasoning Failure**: System fails to correctly synthesize information across multiple retrieved documents
- **Safety Failure**: System produces output that is bias, harmful or policy-violating

## RAG Success Metrics
The metrics can serve as [[Rubric Based Evaluation|rubrics]] and evaluated by an [[LLM-as-a-Judge|LLM-as-judge]]:
- **context recall**: How much of the relevant context was retrieved?
- **context precision**: How much of the retrieved context was relevant?
- **faithfulness**: Is the answer supported by the retrieved context?
- **answer relevance**: Does answer address original task?
- **hallucination**: Does the answer contain objectively false claims (e.g. if context is untrue)?
##
Associations:
- [[z> MOC AI Evaluation]]
- [[RAG]]
- [[AI Evaluation Metrics]]
- [[Golden Evaluation Dataset]]
- [[Rubric Based Evaluation]]
- [[LLM-as-a-Judge]]

Tags: #📥️ 
Sources:
-  [https://www.freecodecamp.org/news/ai-evaluation-engineering-build-a-production-grade-llm-evaluation-platform-handbook](https://www.freecodecamp.org/news/ai-evaluation-engineering-build-a-production-grade-llm-evaluation-platform-handbook/#heading-part-4-rag-evaluation-the-six-metrics-that-carry-all-the-diagnostic-weight)
