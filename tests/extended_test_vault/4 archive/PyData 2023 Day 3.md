## Monorepo
- Deployment: Single Process vs. Micro Services
- Development: Integrated vs. Modular
- Packaging: Builtin vs. Plugins

- Monolith Advantages:
	- shared understanding because code is central
	- easier refactoring and deployment
	- easier CI/CD
- Disadvantages:
	- Different release cycles
	- Different performance profiles
	- Affected changes tracking: don't want to run gazillion tests when only a small part changed
- When to use mono repo?
	- same team working on most parts of project
	- tight coupling of projects
- How to structure
	- Different sub directories for concerns
	- Different packages for concerns
- Affected changes tracking
	- use CI/CD
	- `nx` for monorepo management

## Synthetic Data
- PD - Berater der Öffentlichen Hand
- synthetic data removes identifyable personal information from data, while maintaining its statistical properties
- Usable for:
	- testing
	- for predictive analysis/training
	- alternative for real data
- How to generate synthetic data?
	- simulations
	- Generative ML
		- GANs (real useful, since only D model ever saw original data, never G)
		- VAE
- Evaluation:
	- Resemblance Analysis
	- Utility (based on downstream task)
	- Privacy Evaluation
	- Overall Metrics
- Package: [SDV](https://docs.sdv.dev/sdv/)

## Dynamic pricing at FlixBus
- change price based on properties, instead of just offering a fixed one
	- more profit
	- demand management
- Challenges:
	- Handling large amount of data
	- Optimizing prices in real time
- Large amount of data:
	- Use Kafka to consume data
		- past booking data
		- networking related data
		- predictions about demand
	- Events from Kafka go to db
		- continuous removal of stale events from db
	- k8s horizontal pod autoscaling

## ApacheArrow
- standardized, in-memory format for columnar data
- efficient for in-memory processing

## Feedback for PR
- Blog: https://jdkandersson.com/2022/12/25/giving-and-receiving-great-feedback-through-prs/
- Giving feedback:
	- doesn't have to be acted on
	- keep standard high, for feedback to not get lost
	- be gentle, people connect their identity to code they write
	- Poor feedback:
		- don't focus on the person, focus on the code
		- be precise in feedback, and offer a way forward
- Receiving feedback:
	- assume helpful intent
	- respond with reasoning
	- blind adoption of feedback is as bad as ignoring it
- Checklist:
	- also give positive feedback
	- avoid: "you", "your", "I"
	- use liberally: "seems", "could"
	- prompt others for their opinion after giving yours
	- Keep in mind that you might be wrong
	- Accept progress over perfection
	- be patient, people advance their thinking with a little distance

## Mutation testing
- start with production code, apply mutation to it --> mutant
- packages: [https://mutmut.readthedocs.io/en/latest/](mutmut), [cosmic ray](https://cosmic-ray.readthedocs.io/en/latest/)
	- packages are deterministic
	- try to produce sensible mutations based on some heuristics
- use packages to automatically mutate your code base, and see which mutations do not cause failing tests
- helper for good coverage and good quality tests (testing the tests)
---
Associations: [[z> MOC PyData 2023]]
Tags: #📥️
