### Handling non-determinism
- pass@k
	- rate of getting at least one correct solution in k trials
	- score rises with higher k
		- 50% pass@1: half the tasks solved at first trial
		- 50% pass@3: running three trials correct solutions found for half the tasks
		- e.g. given 50% per trial success rate, trial@1 is 50% and trial@2 is 75%
- pass\^k
	- rate of getting correct answer in all k trials
	- score falls with higher k:
		- e.g. given a 75% per-trial success rate and 3 trials, pass\^3 is 0.75^3=42%

![[RAG Evaluation#RAG Success Metrics]]
##
Associations:
- [[z> MOC AI Evaluation]]
- [[AI Evaluation Fundamentals]]
- [[Evaluation Methods]]
- [[LLM-as-a-Judge]]
- [[RAG Evaluation]]

Tags: #📥️
Sources:
- https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- https://www.freecodecamp.org/news/ai-evaluation-engineering-build-a-production-grade-llm-evaluation-platform-handbook/
- https://mlflow.org/articles/integrating-evaluation-into-ai-workflows-2026-guide/
