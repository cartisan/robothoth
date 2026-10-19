# General
slido, pyber
* Pointers:
	* numba: JIT translating python code into optimized machine code
	* Polyaxon: ??? 

# Streamlit dashbord for production
* model entering use requires trust from user
* user needs access to all kinds of metrics and graphs to build trust
* doesn't have to be difficult, streamlit dashbord is enough
* yay, streamlit.

# Experiment tracking with streamlit
* polyaxon running on kubernetes (whats that)
* use a lot of mlflow

* DVC -  command line tool
	* data versioning
		* replaces large file you dont want to track with git with small meta data files
		* support for data versioning via remote data storing (like git for large files)
	* reproducible pipeline
		* definition via yaml file, declaring depedency files and output files for each stage
		* results in a DAG
		* automatically tracks input and output data with DVC
		* does some caching
	* experiment managing
		* stores each experiment in git via additional commits/tags
		* can show which experiment comes from which git commit
		* can show metrics for each experiment
		* not logging system, its experiments versioning inside code
	* has python API, just like github python api
		* Cool combination: can load data from any commit, and experimen from any commit, and combine them for instance to see on which data two previous models disagree --> Model diffing
	*  integrates with kubernetes and probably with CI/CD setup

* streamlit - dashbord for data apps
	* had his slides in streamlit
	* can be combines with DVC to visualize tracked experiments
	* feels more pythonic than dash, no need to understand callbacks and stuff, wheras dash requires understanding of WebDev practices

* Pros and Cons
	* Pros
		* Can do whatever you want in your UI
		* develops with the project and is versioned
		* fast prototyping of dashbord
	* Cons
		* recent projects, so perhaps not stable/not mature
		* harder to maintain in long term: ML team has to maintain it
			* technical debt
			* have to maintain compatibility with older pipeline

# Clean code
* clean code reads almost like prose
* formatters are good
* KISS

# Deploying ML models using ONNX 
* scikit learn deployment: just pickle the model and its done
* Deployment: how hard is it?
	1. load pickle in web service
		* pickles are not self contained, so need to manually reproduce all training dependencies in production (including versions) = coupling training and deployment
	2. Send request to web services
		* Latency on single rows is too high, for a model that is trained on batches
	3. Interoperabiity with rest of tech stackother other 
* ONNX
	* a standardized way to store ML models in binary format that seperates training from prediction
	* growing ecosystem
	* backed by Microsoft
	* Details:
		* represents models as DAGs, nodes are operators, edges are typed arrays
		* 175 available operators
		* stores DAG and paramters in binary format
* Converting models to ONNX
	* entire pipline including preprocessing can be converted to ONNX **iff** you can represent it as DAG of ONNY operatirs
	* converters exist for established ML frameworks (e.g. sklearn-onnx)
* Inference in Deployment
	* only need onnxruntime as dependency (no matter if model was TF, torch, or Java)
	* create a session object from stored ONNX model, and run it
	* provides a ~10x/100x speedup in inference compared to sklearn, because its 

# Modle drift
* standard monitoring is not enough
	* ground truth are delayed
	* past performance doesnt guarantee new results
	* target function is volatile
* data quality
	* misisng values, domain knowledge, constants
	* reference data: descriptive statistics, data distribution type
		* point estimation as simple solution
		* statistical test to get confidence intervals
* Evidently
	* has data quality dashboard
	* important to define: statistical tests, thresholds and stat test paramaters, feature importance
* approaches for drift detection
	* parametric tests
 		* approaches:
			* t-tests for means (one/two samples)
		* more sensitive to drift than non-parametric tests
		* need specific test for each feature
	* non-parametric tests
		* approaches:
			* Kolmogorov-smirnov test
			* Wasserstein distance
			* chi-squared test
			* ... (there is more)
		* simple heuristic for test choice based on feature type
		* less sensitive to drift
* large datasets
	* tests can be too sensitive, since statistics are meant for (small) samples and not whole population
	* practical solutions:
		* sampling (pick representatives datapoints using some strategy)
		* bucketing (aggrgegate datapoints)
* non-batch models
	* calculate metrics continuously or incrementaly using windows
* nuances
	* data drift: detected, feature drift: not detected
	* positive interpretation:
		* important feature did not change
		* model is robust
	* negative interpretation:
		* our model is bad, since change in important features desnt change prediction
* dirft detected, what next:
	* run data quality, before acting on model predictions
		* solve data quality issues
	* prediction and data drift:
		* interprete this, somtimes things just change
		* (see image)

# Honey, I shrunk the target variable
* blog under: inovex.de/de/blog/honey_i_....
* RMSE is most often used metric, but not necessarily always the best
* RMSPE: root mean squared percentage error
* MAPE: mean absolute percentage error
* **Minimizing (R)MSE is actually estimating the mean of the residual distribution**
* Minimizing RMSE with log(price)
	* take log(price)
	* minimize for MSE
	* transform back
	* --> we get the median (to transform for mean, add 0.5*\sigma² )
* **Transforming the variable might mean transforming what you are minimising for**
* **Applying a logarithm means going from absolute relations into percentage relations**

# Grokking LIME
* Github: ionicsolutions has the notebook
* Four principles of XAI (NIST) -> Philips et al. 2021
	* Explanation: Something that delivers or contains accompanying evidence or reasons for outputs or processes
	* Menaingful: Provided explanations are understandable to endusers
	* Accuracy: Explanations correctly reflect reasons for generated ouputs, correctly represent execuetd processes
	* Knowlege Limits: System only operates when it reaches sufficient confidence for its predictions

* LIME Local Interpretable Model-Agnostic Explanations (Ribeiro et al. 2016)
	1. Segment the image (e.g. felzenszwalb)
	2. Generate samples (each a list of segment_ids with 1/3 of the segments set to 0 and rest to 1)
	3. Generate images from samples (with pixels associated with missing segments set to 0)
	4. Make predictions on generated images
	5. Fit linear model (predicting which segments have which influence on prediction)
	6. Generate visual explanation (show important segments)

* Facebook AI has a library with XAI algos for pyTorch
---
Associations: [[z> MOC PyData 2022]]
Tags: #📥️ (extract ideas from workshop notes)
