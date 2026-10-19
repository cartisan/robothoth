## Calibration
Calibrate LLM with domain expert judgement:
1. Build dataset of real outcomes or traces, including expert grades (and perhaps annotation)
2. Divide dataset into
	- 20% as examples for judge's context
	- 40% for test-set
	- 40% for final validation set -- prevent overfitting
3. Develop judge's prompt iteratively, using test set
	- If expert and LLM do not align, change prompt or context
		- *Rubric problem*: Is a criterion missing to represent human's evaluation?
		- *Edge case*: Is an example missing to capture this case?
	- Compare answers of LLM and SME using e.g. Spearman's rank correlation
## Biases
LLM judges suffer from common types of biases, that should be accounted for in the evaluation system:
- Position bias
	- When comparing two answers, LLMs prefer first answer over second
	- Solution: Run each comparison twice, changing order
- Verbosity bias
	- LLMs tend to prefer longer answers
	- Solution: Add concision criterion into grading sheet
- Self-Preference bias
	- LLMs tend to perform answers from own model family
	- Solution: either only compare same family, or use independent family for judge
## G-Eval Approach
- Force judge to write out step by step CoT
	- logically constraints output of LLM to its own reasoning
- Benefit: often leads to higher agreement rates with humans
- Drawback: higher cost per eval run
##
Associations:
- [[z> MOC AI Evaluation]]
- [[Evaluation Methods]]
- [[Rubric Based Evaluation]]
- [[Golden Evaluation Dataset]]
- [[AI Evaluation Metrics]]
Tags: #📥️ 
Sources: https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80
