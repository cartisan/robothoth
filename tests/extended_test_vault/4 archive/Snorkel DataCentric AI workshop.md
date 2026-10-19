## What Can Data-Centric AI Learn From Data & ML Engineering?
### Guiding Principles for Monitoring
* Prioritize human attention
    * reduce falce positives
    * alert should provide context
* Deep Monitoring
	* Monitor multiples points of the pipeline, not just the endpoints
* Automation
	tag:#📥️* suggest next steps or slices to check
	* auto-rollback to last version

### Have Code Centric Workflow
* not model centric workflow
* DS creates code, hands off to ML Engineer who automates model training using that code continiusly
* means most steps (like data collection) need to be in code

### Versioning
* use end-to-end versioning: not just code, also models, data, metrics, results


# Monitoring
- Training: 
	- dataset, transformation, training, evaluation
	- use MLMetdadata, preserves lineage of artifacts > entire history of model
- Serving:
	- Operational Performane: Latency, Scaling Up, Scaling Down
	- Capturing Prediction Requests made to model --> collect your new dataset

# Datasets
- store metadata and lineage of data
 
---
Associations:
Tags: #📥️ (extract ideas from workshop notes)
