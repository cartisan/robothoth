The dataset is more important than metrics:
- mediocre metrics on good data will catch more than good metrics on poor data
- data defines space of problems you evaluate, metric define precision of that measurement
## Properties of good datasets
- **representative**: 
	- reflects actual distribution of user input, not idealized version in programmers mind
	- has: *edge-cases*,  *adversarial examples*, *samples from long-tail*, and uses *domain-specific terminology* 
	- consider running a harvester that automatically finds production cases that should go into the golden dataset (e.g. high latency, negative user feedback, ...)  
- **labelled**:
	- each case has ground truth annotation that SMEs would agree solves it
	- can be single answer or set of criteria that an answer has to fulfill (see [[Rubric Based Evaluation]])
- **versioned**:
	- the dataset will evolved to cover new failure cases, so version it and keep a *changelog explaining why a case was added*
- follows a **clear schema**:
	- keep the data contained in a case consistent
	- e.g: 
		- metadata: id, dataset version, added_reason, failure_mode
		- input: query, conversation_history, expected_context
		- ground_truth: ideal_answer, answer_criteria
	-  failure mode is most important, forces person adding a case explicate what it should catch

![[AI Evaluation System Roadmap#1. Collect tasks for the initial eval dataset]]

##
Associations:
- [[z> MOC AI Evaluation]]
- [[AI Evaluation Fundamentals]]
- [[Rubric Based Evaluation]]
- [[AI Evaluation System Roadmap]]

Tags: #📥️ 
Sources:
