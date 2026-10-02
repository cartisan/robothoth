- Analyze
	- review 100 user traces
	- add manual notes (open coding)
	- find problem patterns in notes (axial coding)
		- measure frequency of each failure mode
- Measure
	- for deterministic failures, build code based evals aka tests
		- e.g. different ways of telling LLM a date and expected date
	- for subjective failures LLM as judge
		- e.g. when should LLM hand off to human
			- real conversation traces, human pass/fail and critique by human
			- train judge on 20% of this data, develop prompt by evaluating on 40% of data, 40% as final evaluation holdout dataset to guard against overfitting
			- iterate on dev set:
				- where LLM and human misaligned, change prompt
					- rubric problem: a category missing in prompt
					- edge case: specific example not handled well
- Automate
	- Use CI/CD 
	- tests can ran regularly, LLM as judge more rarely e.g. nightly
- Improve
	- use production monitoring
		- sample production traces: random as well as detected failures
		- review sampled traces (subset)
		- repeat error analysis, to find next issue to solve
##
Associations: [[AI Evaluation System Roadmap]]
Tags: #📥️ 
Sources: https://newsletter.pragmaticengineer.com/p/evals
![[The Pragmatic Engineer - A pragmatic guide to LLM evals for devs.pdf]]