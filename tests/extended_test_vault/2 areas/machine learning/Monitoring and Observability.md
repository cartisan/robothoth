- Definitions:
	- Monitoring: Catching problematic behavior
	- Observability: Understanding what went wrong
- Principles:
	- Should cover all parts of the model: output, internal functioning and input
		- output: predictions, their distribution and training quality
		- internals: created and cleaned feature, their distributions and quality
		- input: data health
		- general advice: start monitoring from the end, i.e. output monitoring first
	- Categories of monitoring
		- to notify: observations that should trigger an alert because they indicate sever degradation
		- to monitor: observations that give an overview over general health and should be watched regularly
		- to analyze: observations that help investigate incidents and drill down on problems
	- Monitor snapshot and development
		- some dashbords should show an overview of the current state as comprehnsive as possible
		- other dashords should show an overview of the development of relevant metrics over time

- Four Golden Signals
	- ...

## Resources:
- [Zalando's CTO PyCon Keynote on ML Monitoring on Youtube](https://www.youtube.com/watch?v=QcevzK9ZuDg) 
- Also book: Chip Huyen, Desgning ML Systems (recommended at PyCon 23)
- Evidently Observability Course:
	- [[Observability Course - Intro]]
	- [[Observability Course - Monitoring Metrics]]
	- [[Observability Course - Designing Effective Monitoring]]
---
Associations: [[default_vault/0 inbox/z> MOC ML]]
Tags: #🌱
