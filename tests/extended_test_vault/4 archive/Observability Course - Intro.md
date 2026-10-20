## What can go wrong
- Training-serving skew
- Problems with data quality and integrity (preprocessing issues, schema changes, data loss)
- Broken upstream model
- gradual or sudden concept drifts: something happens in the problem domain
- data drift: data distributions change
## What is ML monitoring and observability
- Monitoring: track and analyze performance of ML models in production
	- domains
		- data health
		- model health
	- Difficulties:
		- what is right metric?
		- can we get necessary (labeled) data?
	- ℹ️ more reactive (known unknowns): tracks a set of metrics, helps detect issues
- Observability: allow inspection of all parts of deployed system
	- ℹ️ more proactive (unknown unknowns): gives visibility to system behavior, help understand causes (why is something happening, where is it happening)
## Types of evaluations
- ML system is not just a model: 
	- Business KPIs: e.g. number of purchases, loan approval rate, ...
	- Model quality: precision+recall, MAE, top-k accuracy; also: prediction drift, input data drift
	- Data quality: missing values, type mismatch, range violations, ...
	- Software system health: latency, error rate, memory usage, ...
## Key considerations when setting up
- Service implementation: real time, batch, ad hoc script?
- Feedback loop: immediate or delayed?
- Environment stability: stable or fast changing?
- Service criticality: business cost of a quality drop / error?
- Model retraining approach (couple monitoring and retraining?)
	- How expensive is retraining?
	- Retraining on trigger or schedule?
	- External boundaries for updating models (regulations, approval process...)
- Reference dataset
	- compare incoming data with selected data set showing all expected data patterns and where model works well
	- compare each batch of data with reference data set
## Architecture overview
- Batch: 
	- run monitoring job on a schedule (e.g. every hour) or a trigger (e.g. new label arrives)
	- works also when ML system is online, in that case analyze db or logs
- Real time:
	- send data from ML service to monitoring system
	- compute metrics in real time and store in time series db to show in a dashboard
- Batch ML serving with delayed ground truth
	- have ML pipelines run: data validation, prediction, training
	- after each run execute relevant checks: input data checks, model prediction checks, model quality checks
	- log results to db and display in monitoring dashbords ![[Pasted image 20231018142029.png]]
- Online ML service
	- each request to ML system send inputs and predictions to ML monitoring service
	- ML monitoring service computes ongoing/running metrics and logs results to db
	-  Use some sort of time series db like prometheus ![[Pasted image 20231018142302.png]]
- Frontend
	- no user interface: just read logs or db, can already include alerting to e.g. Slack
	- one off reports: no real time feedback, but you can pull information
	- Reuse existing BI service monitoring
	- Dedicated ML monitoring dashbord

---
Associations: [[z> MOC Evidently Observability Course]]
Tags: #📥️
