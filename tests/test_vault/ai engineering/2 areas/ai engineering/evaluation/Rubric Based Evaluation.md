- Define multiple criteria that an outcome must meet
- Use [[LLM-as-a-Judge]] to evaluate the outcomes based on the criteria using a Likert scale
- Provide anchor examples for extreme ends of the Likert scale

## Properties of good criteria
- **Specificity**: Provides a clear definition of success
	- Instead of: "Is this a good answer" -> "Does the answer provide actionable steps relevant to the original question"
- **Measurability**: Must be inferrable from the output of a system alone (context + reference answer)
	- E.g. "Did the answer provide at least three distinct solutions"
- **Independence**: Criteria should not overlap
	- E.g. if grammar and typing scores correlate often, extract them into a single fluency criterion

## Example
Airline deploying AI Chatbot to treat customer booking complaints
- Criteria:
	- *Politeness*: "Is the tone is empathetic to the user’s frustration, does it maintain professional distance, and avoid robotic, repetitive phrasing?"
	- *Policy-Adherence*: "Did the answer refuse the refund request based on the delay policy?"
	- *Escalation-Correctness*: "Did the agent hand over the conversation after three tries of solving the customers request?"
## 
Associations:
- [[z> MOC AI Evaluation]]
- [[Golden Evaluation Dataset]]
- [[Evaluation Methods]]
- [[LLM-as-a-Judge]]
Tags: #📥️ 
Sources: https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80
