## Demo Notes
- have metric tracking, as well as model and data set versioning <-- experimentation
- have model managing and deployment tools, as well as infa scaling <-- deployment
- just a python library: `wandb`
- experiment tracking
	- like Mlflow: run `wandb.init` and `wandb.metrics` --> logs metrics to local/remote server
	- supports OpenAI, Hugging Face, sklearn, PyTorch, tensorflow, ... via integrations
	- can log metrics and artifacts like images, also saves metdata (like who ran and which script, but also things like ressource-usage or precise pip environment used to run job)
	- workspace might be messy, but projects can have reports that surface relevant informations and metrics
- model registry
	- lineare of runs and datasets creating models
	- give models created by experiments aliases
	- have automations for models in registry
		- executes specific tasks in certain events, e.g.
			- run CI/CD validation step via web hook when new model version is registered to create a comparison report
			- when tagging a model as production it deploys the model to prod
		- can out rules and rights on automations:
			- only allowed to tag model as production by an ML Engineer after model validation was executed
- courses: www.wandb.courses
	- Effective MLOps: CI/CS for ML (Gitops)
	- Effective MLOps: Model Development
## Work-practice considerations
- Consider whether experiment tracking can improve reproducibility of routine work.
- Review model lineage, compute metadata, and permissions before choosing automations.
- Keep a clear distinction between recording an artifact and approving its deployment.

---
Associations: [[z> MOC Infrastructure]]
