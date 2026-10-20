## Approaches

| Evaluation Type                | Methods                                                                                                                | Strength                                                                           | Weaknesses                                                                                     |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Code-based**                 | - String match checks<br>- Unit tests<br>- Outcome verification<br>- Tool call verification<br>- Transcript analysis   | - Deterministic<br>- Fast and Cheap<br>- Easy to debug                             | - Brittle to valid but unexpected variations<br>- Hard for subjective tasks                    |
| **Model-based (LLM as judge)** | - [[Rubric Based Evaluation]]<br>- Natural language assertions<br>- Reference-based evaluation<br>- Multi-judge consensus | - Flexible<br>- Captures nuance and open-ended tasks<br>- Handles free form output | - Non-deterministic<br>- More expensive than code<br>- Requires human grader based calibration |
| **Human**                      | - SME (subject-matter-expert) review<br>- Crowdsourced judgemend<br>- A/B testing<br>- Inter-annotator agreement       | - Gold standard quality<br>- Used to calibrate model based graders                 | - Expensive and slow<br>- Access to experts at scale                                           |

## Libraries
- [RAGAS](https://docs.ragas.io/en/stable/): python library for evaluation loops
- [DeepEval](https://deepeval.com/): basically pytest for LLMs, with CI/CD support

##
Associations:
- [[z> MOC AI Evaluation]]
- [[AI Evaluation Fundamentals]]
- [[LLM-as-a-Judge]]
- [[Rubric Based Evaluation]]
- [[AI Evaluation System Roadmap]]

Tags: #📥️
Sources:
- https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- https://www.freecodecamp.org/news/ai-evaluation-engineering-build-a-production-grade-llm-evaluation-platform-handbook/
- https://mlflow.org/articles/integrating-evaluation-into-ai-workflows-2026-guide/
