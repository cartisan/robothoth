# General
* Continuous Machine Learning: CML (fast machines for your CI/CD)

# Synthetic Data
## From Real Data
* Train a generative model on real data (like GAN)
* benefit

## From assupmtions
* Use background knowledge
	* Statistical distributions or historic trends
* Use existing models
	* simulations or statistical models
* Helpful with limited historical data, novel situation, expert analyst

## Types of synthetic data
* generated from real, (non-)public data
* Generated from existing knowledge
* Based on analyst knowledge
* Generated from generic assumptions

## How to measure realness
* comparing univariate statistics
	* plot each feature of real and synthetic data against each other
	* measure Hellinger distance
* comparing bi-variate statistics
	* compute correlations (like Pearson's)
* distinguishability
	* train binary classification model to predict wheather data is real or synthetic

## Frameworks
* Pydbgen
* Faker
* Trumania
* DataSynthesizer
* SyntheticDataValut: GANs, generative distributions functions
* TimeseriesGenerator <-- !!

# Pandas SettingWithCopyWarning
* copy vs view
	* view: shares data with origin (pointer)
	* copy: new data without pointer
* Problem: not clear which operations return a copy, and which return a view
* sloutions currently:
	* use a lot of .copy()
	* use .loc instead of []

# XAI meets NLP
* Larissa Haas
* Types
	* Global vs. Lokal: Explains effect on whole dataset, local only on single prediction
	* Model specific vs. Model Agnostic: care about implementation or not
## Approaches
* Human annotations 
	* include reasons for each label, model learns to also predict reasons apart from label
* Feature importances and model structures
	* Observe attention of layers to certain inputs (e.g. words)
* Use interpretable models like decision trees
	* ... duh
* LIME, SHAP & ELI5
	* LIME: see effect of local area (like image segment) on prediction
	* SHAP: also works globally
	* ELI5: Explain like I'm 5
		* works out of the box on scikit learn models
* Graphical Representations
	* 2D representation of embeddings 

# Make MLOps Uncool again
## Goals: 
* Reproducibility
	* Config
	* Data
	* Model
	* Pipeline
* Automation
	* Train, Eval, deploy

## Approaches
* End to End ML Platforms
	* vendor lockin
	* scalability issues (overkill sometimes)
* OSS
	* tooooo many tools, unclear how to connect
* DevOps Tool
	* Kubeflow... its nice
	* Git doesnt work because you need to version data, models and params
* Tutorial: https://github.com/iterative/workshop-uncool-mlops

---
Associations: [[z> MOC PyData 2022]]
Tags: #📥️ (extract ideas from workshop notes)
