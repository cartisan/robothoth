## Intro
- standard model performance metrics exist
	- model quality and error metrics: F1, log-loss, MAE, ...
	- business metrics: purchases, clicks,...
- but problems:
	- Past performance does not guarantee future results, in unstable environment
	- delayed ground truth
	- volatile target function where local drops occur that do not indicate performance loss
	- many segments with different quality, so aggregated metrics are not enough
- solutions: early monitoring
	- data quality: integrity and quality changes
	- data drift: change in input data distributions
	- output drift: changes in prediction distribution

## Model quality
- quality degradation happens in time, each retraining increases quality again
	- need to continue model performance continuously in prod (but labeled data required for that)
- Classification quality metrics
	- accuracy, precision, recall, F1, ROC AUC, log loss
- Regression quality metrics
	- MAE, RMSE, MAPE, SMAPE (symmetric MAPE)
- ranking metrics
	- cumulative gain (CG): sum of value of selected objects (ignores order of selected objects)
	- precision/recall/lift @k:
		- precision and recall of relevant objects in top k results
		- lift: improvement over random ranking - precision at k of our model / precison at k of random
	- serendipity, novelty, diversity, coverage, popularity bias
- production monitoring
	- pick right metric (dough)
	- pick right aggregation window (e.g. precision over sliding 7d window)
	- consider splitting data into segments if segments vary by performance

## Data quality
- data processing issues
	- wrong source
	- lost access
	- bad sql
	- broken feature code
	- data schema changes
	- broken upstream models
- what to do
	- data profiling: statistics and distributions for data, comparing with thresholds or reference data
	- General data quality properties (no reference data):
		- low share of missing values
		- do duplicate rows/columns
		- no near constant features
		- no highly correlated features
		- no target leaks (high correlation between feature and target)
		- no range violations
	- Compare current batch of data with reference (e.g. training) data
		- expected schema, completeness, batch sizes
		- expected patterns on columns
			- specific distributions, ranges, descriptive statistics
		--> possible to auto generate test conditions from reference data (e.g. by evidently)

## Data and Prediction Drift
- prediction drift
	- compare model output distribution on training data and current data
- feature drift
	- same for individual features
	- perhaps necessary to aggregate, like: share of drifting features
	- less important than prediction drift
- not all data drifts are important
	- many weak features: we don't care
	- this is only a heuristic we need due to delayed feedback (i.e. ground truth) anyway
- how to detect drifts
	- method
		- statistical tests: parametric and non-parametric
			- more sensitive
			- best for smaller dataset 
		- distance metrics: Wasserstein distance or population stability index
			- better for larger datasets
		- rule based checks:
			- great when you expect particular changes
	- threshold:
	- reference dataset:
	- alert conditions:
		- top features drifting
		- share of features
- tips
	- start with data quality checks, thats where most problems lies
	- distinguish between feature set: if you have few important features alert differently than when you have many weak ones
	- monitor per segments, not in aggergate
### Parametric tests
- z and t tests (one or two sample tests)
- one samples tests require different tests per feature: need to come up with expectations 
	- e.g. normally distributed with mean X
- two sample tests require reference data set
- more sensitive than non-parametric tests
- hard to fine tune for lots of features
--> use with few interpretable features
### Non-parametric tests
- less sensitive to drift
- less demanding, can use heuristic to choose  test based on data type
- Kolmogorov-Smirnov test
	- distirbution equality for continous data
- Pearsons Chi² test
	- distribution equality for categorical data
- K Sample Anderson-Darling test
	- can several collections of obs be modeled as coming from one pop
- Fisher's exact test for small samples
### Distance based
- any metric that compares distributions
- easier to pick interpretabel metric than statistical test
- makes sense for large data sets
- Wasserstein distance
	- how much effort to turn one distribution into another
- Jensen-Shannon divergence
	- distance between prob distributions
- Use binary classification algorithm to predict whether samples come from ref dataset or current dataset, see how good it gets
### Special cases
- large datasets: statistical tests were made to work on statistical samples, not the whole population
	- statistical tests will give a lot of false positives in this case
	- pick a sample of representative observations or aggregate over several observations (e.g. drift detection on minutes instead of seconds) before applying statistical tests
- non-batch models: continous data stream instead of batches
	- use windowing (right size and window function needs to be selected experimentally)
	- compare windows with each other to detect drifts
- drift detection as retraining trigger
	- not recommended: data might be low quality or not sufficient, data might be non-representative
	- instead: use drift detection as investigation trigger, or use a strong model training evaluation process before roll out
### Interpreting drifts
- Data drift: yes  prediction drift: no
	- positive case:
		- important features did not change
		- model is robust enough against drifts, yay
	- Negativ case:
		- important features changed but model predictions did not
		- model has problems extrapolating, you should intervene
- Data drift: yes  prediction drift: yes
	- positive case
		- important features changed and model reacts reasonably
		- model extrapolates well, yay
	- negative case
		- important features changed and model reacts unreasonably, need to respond

## What to do when drift is detected
1. check data, to make sure drift is "real" (and not data source/quality problem)
2. understand drift: check what features have changed and how much
	  - distributions changes
	  - correlation changes
	  - descriptive statistics
	  - also check all segments of data (did individual cements change?)
	  - Is there a real world interpretation?
3. Doing nothing is an option
	- false positive, or model adapts well
	- wait for retraining to solve problem
	- monitor the development closely
4. Respond
	- retrain model
	- update model with new features / fixes to make it more robust
	- tune model hyperparameters

---
Associations: [[z> MOC Evidently Observability Course]]
Tags: #📥️
