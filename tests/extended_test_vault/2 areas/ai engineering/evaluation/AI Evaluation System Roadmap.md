### 1. Collect tasks for the initial eval [[Golden Evaluation Dataset|dataset]]
1. **Start early with small sets**: Begin with 20–50 tasks derived from real failures rather than waiting for hundreds; early system changes yield large effect sizes that small samples can easily catch.
2. **Convert manual tests and bug reports**: Use existing manual pre-release checks, support tickets, and bug trackers to build realistic, high-impact test cases.
3. **Ensure clear specs and reference solutions**: 
	- Draft unambiguous tasks with explicit pass/fail criteria verified by domain experts
	- Write a working reference solution to prove task feasibility and validate graders
4. **Build balanced, non-biased problem sets**: Include both positive and negative cases (e.g., when to trigger a web-search action versus when to rely on pre-existing knowledge) to prevent single-sided optimization and class imbalance.
### 2. Design the eval harness and graders
1. **Isolate test environments**: Guarantee clean, reproducible state for every trial to avoid cross-run contamination (e.g., leaked git history or cached files) and infrastructure-level correlated failures.
2. **Prioritize outcome-based grading over execution steps**: Grade final outputs rather than exact step sequences or tool-call order to avoid penalizing creative, valid execution paths.
3. **Implement partial credit**: Design multi-component evaluations to capture intermediate progress along a continuum rather than using binary pass/fail mechanics.
4. **[[LLM-as-a-Judge#Calibration|Calibrate]] and isolate LLM-as-a-judge graders**: 
	- Carefully align model graders with human judgment
	- Provide agent with fallback options (e.g., answer "Unknown" when not enough knowledge) to mitigate hallucinations
	- Use separate, single-dimension [[Rubric Based Scoring|rubrics]] with seperate LLM judges where necessary
5. **Harden graders against edge cases and exploits**: Verify that graders handle numeric tolerances and stochastic outputs cleanly, and design tasks so agents cannot bypass constraints or cheat.
### 3. Maintain and use the eval long-term
1. **Regularly inspect trial transcripts**: 
	- Audit raw logs and trace outputs to distinguish true agent failures from broken task specs, flawed harness configurations, or rigid grading.
	- Invest in tooling for fast and easy trace review
2. **Monitor for score saturation**: Rotate or scale up evaluation difficulty as scores near 100% (pass rates saturation hides capability gains on harder tasks and limits regression testing).
3. **Adopt Eval-Driven Development (EDD) and shared ownership**: 
	- Treat eval suites as living test-codebasses with centralized core infrastructure
	- Enable product teams, domain experts, and non-engineers to contribute tasks via PRs
	- Try to work eval driven, i.e. building evals out before writing feature code
### 4. Implement Production Monitoring
- Create a system to sample and evaluate (some) traces from production
- Next to metrics also measure latency and cost
* Goals:
	- **Detect distribution shifts**: changes in topics, phrasing or failure modes
	- **Mine new eval cases**: Identify low-quality traces, label them, add them to eval set
	- **Validate model updates**: performance on offline eval does not need to match

## 
Associations:
- [[z> MOC AI Evaluation]]
- [[AI Evaluation Fundamentals]]
- [[Golden Evaluation Dataset]]
- [[Evaluation Methods]]

Tags: #📥️
Sources:
- https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- https://www.freecodecamp.org/news/ai-evaluation-engineering-build-a-production-grade-llm-evaluation-platform-handbook/
- https://mlflow.org/articles/integrating-evaluation-into-ai-workflows-2026-guide/
