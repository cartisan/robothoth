## Overview
- Questions when setting up an ML monitoring system
	- How to select and prioritize ML monitoring metrics.
	- How and when to retrain ML models. 
	- How to choose a reference dataset. 
	- How to implement custom metrics in ML monitoring.
	- How to choose an appropriate ML monitoring architecture.

- What is a good ML monitoring system:
	- Instrumentation: useful metrics and statistics for further analysis
	- Alerting: define unexpected behaviors we can detect and action policies for them
	- Debugging: provide enough context to resolve issues and understand model behavior
## General Architecture
- What to log:
	- For each prediction events, log all relevant data into prediction store
		- model inputs
		- predictions
		- (if available) ground truth
- Two ways for monitoring:
	- Service to which predictions are also sent, apart from prediction store
		- computes metrics and sends them to a metric store
	- A regular monitoring job that pulls predictions from the prediction store
		- computes metrics and sends them to a metric store
	- A monitoring dashboard that queries the metric store
	- With smaller systems, you might not need a metric store and the monitoring dashboard can just talk to the prediction store (but: doesn't scale well, since generally different type of function is required for prediction store and metric store)
## What to log
- Focus on alerting: only key metrics
- Focus on debugging: metrics and more data like descriptive statistics
- Priority:
	1. Symptom: e.g. 1d accuracy
	2. Summary metrics: e.g. TP, FP, predicition volume, % nulls
	3. input/output performance infos: e.g. per-column statistics, distributions,  segmented model performance
	4. Debugging + analytics data: correlations, explainability
	5. row-level query: queries that log objects/rows of interest
- Depth:
	- minimalistic monitoring: few models, small datasets, monitoring for technical users, easily accessible predictions
	- in-depth observability: many models, large datasets, critical use cases, monitoring for business stakeholders
- Service health:
	- does it work: like golden signals -  traffic, latency, ...
	- model performance: how does it work - business KPI and direct model quality
		- get alerts!
		- if you want to enbale debugging, log distributions of training data or errors per bucket (e.g. per cement error)
	- Data quality and integrity: 
		- track data health metrics 
		- run a manual data test suit and track number of failing tests over time
	- Data and concept drift: is model still relevant?
		- track relevant metrics
		- log distributions to allow debugging


---
Associations: [[z> MOC Evidently Observability Course]]
Tags: #📥️
