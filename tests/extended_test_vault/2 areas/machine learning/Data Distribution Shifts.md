## Concepts
- *data drift*: distribution of inference data changes over time, until distribution of training data does not match it anymore
	- covariate shift happens, when features are differently distributed in training and inference data, i.e. `P(X)` changes but `P(Y|X)` stays the same
	- label shift happens, when labels are differently distributed in training and inference data, i.e. `P(Y)` changes but `P(X|Y)` stays the same
	- Example: Given a classification problem of detecting cancer, age is a relevant feature
		- You have more people with age > 40 in your training than in your inference data, so `P(X)` is different, but the probability that a patient has cancer when they are over 40 is the same in both data sets so `P(Y|X)` is the same --> covariate shift
		- Since there are more people > 40 in the training data, then in the inference data, the percentage of patients with cancer is higher in the former so `P(Y)` is different, but the probability that a given cancer patient is > 40 is the same in both data sets so `P(X|Y)` is the same --> label shift
		- Both shifts can appear at the same time, but do not have to

- *concept drift*: underlying properties of the domain change, e.g. which features and values are relevant to identify a certain concept
	- that is the conditional distribution `P(Y|X)` changes, while the input distribution `P(X)` stays the same
	- Example: Given a regression problem of predicting house prices, a pandemic hits. The distribution of house properties remains the same, but the houses lost value, so `P(Y|X)` has changed

- With data drift, the decision boundary does not change; only the probability distribution of the inputs change. With concept drift, the decision boundary changes.

## Detection
- In general: data distribution shifts are only relevant when model performance degrades, so monitor model performance
- Distributions to monitor: input distribution `P(X)`, label distribution `P(Y)`,  and both conditional distributions
	- in practice, `P(Y)` is often unknown so most industry applications focus solely on `P(X)`
- Comparing distributions
	- compare their statistics like mean, variance, skewness, kurtosis, ...
		- when these metrics differ we have a shift, but when they do not this doesn't mean we do not have the shift (unless we know the type of distribution of our data)
	- perform two-sample hypothesis testing to determine whether the two populations follow the same distribution
		- works better on low-dimensional data
	- compare time scale windows of your data
		- allows detecting temporal shifts
		- the windows we look at affects the shifts we can detect: if we have a periodicity of 7 days in the data, but use a 5 day window this will result in false positives
		- but: the shorter the window, the faster changes can be detected

## Addressing Shifts
- train on huge datasets, in hope of learning a big enough distribution that we don't are about shifts
- adapt trained model to target distribution without requiring new labels: e.g. domain invariant representation learning --> more academic and less practical
- retrain model using data from new distribution
	- either from scratch or fine-tuning
	- big question: which data period to select data from for retraining
- when designing ML systems, remember trade-off between performance and stability
	- i.e. gain due to higher accuracy due to a feature vs. high variation of this feature that might lead to regular data shifts

- ## Resources
	- sklearn drift detection algorithm: [ADWIN](https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.drift_detection.ADWIN.html), Paper: [Learning under Concept Drift](https://arxiv.org/pdf/1010.4784.pdf) 
	- [Alibi Detect](https://github.com/SeldonIO/alibi-detect) python framework

---
Associations: [[default_vault/0 inbox/z> MOC ML]]
Tags: #🌱
