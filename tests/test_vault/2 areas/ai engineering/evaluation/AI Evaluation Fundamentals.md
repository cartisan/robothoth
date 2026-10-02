- Evals make problems and behavioral changes visible before they affect users
- Evals prevent changes from resulting in regressions
- The primary bottleneck limiting reliable AI deployment is poor evaluation methodology, not agent capability.
- Approach:
	- Give agent an input, grade its output
	- Single-turn: prompt -> response -> grading
	- Multi-turn: prompt -> agent modifies environment -> grading
## Structure
- **Task**: single test, with defined inputs and outputs
- **Suite**: collection of tasks, usually sharing a goal (like CS eval suite)
- **Trial**: attempt at solving the task
- **Transcript**: trace: complete record of the trial (reasoning, tool calls, intermediate results...)
- **Outcome**: final state of environment after trial (files, db, ...)
- **Grader**: 
	- evaluates aspect of agents performance, via transcript and outcome
	- task can have multiple graders with multiple assertions
- **Evaluation harness**: whole environment running evals end-to-end

### Types of evals
- Capability:
	- Checking what agents does well
	- Should start with low pass-rate, and then used to navigate hill-climb
- Regression:
	- Does agent still solve tasks it used to
	- Should have near 100% pass rate
	- Old capability evals should become regression evals
### Tiers
- **Offline Evaluation:** Are we improving the status quo?
	- Golden dataset evaluation
	- Component-level evaluation (e.g. retrieval separate from generation)
- **CI/CD Gates**: Is this change safe to ship?
	- Automated on every MR
	- Performs regression testing
- **Online Production Monitoring:** Is the system working correctly right now?
	- Sampling of life traces
	- Distribution shift detection (**?**)
	- Alert on quality degradation
##
Associations:
- [[z> MOC AI Evaluation]]
- [[Golden Evaluation Dataset]]
- [[Evaluation Methods]]
- [[AI Evaluation Metrics]]
- [[AI Evaluation System Roadmap]]

Tags: #📥️
Sources:
- https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- https://www.freecodecamp.org/news/ai-evaluation-engineering-build-a-production-grade-llm-evaluation-platform-handbook/
- https://mlflow.org/articles/integrating-evaluation-into-ai-workflows-2026-guide/
