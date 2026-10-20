- happens in systems where model predictions can influence training data
	- example: job hiring system that learns that "went to Stanford" is an important feature, selecting only CVs from people who went to Stanford, resulting in the reinforcement loop that only people who went to Stanford were hired

- detecting degenerate feedback loops
	- measure metrics for output diversity (like *aggregate diversity* or *average coverage of long tail items*)
	- compare model performance of popular samples against unpopular samples, if unpopular sample performance is bad there is probably a popularity bias in the model
	- track homogeneity of system output over time, if it rises there might be degenerate feedback loops

- correcting feedback loops
	- introduce sufficient amount of randomness in suggestions to users and assign a sampling budget for new samples, so that you get enough unbiased training data for them (increase exploration)
	- check for order effects by introducing position features: e.g. track if a suggestion was clicked when it was the top suggestion, vs. not top suggestion, to be able to estimate importance of placement at the top

---
Associations: [[default_vault/0 inbox/z> MOC ML]]
Tags: #🌱
