# General
slido, pyber
# ML Monitoring talk (Lina Weichbrodt)
* [https://www.youtube.com/watch?v=wWxqnZb-LSk](https://www.youtube.com/watch?v=wWxqnZb-LSk)
* detect fast, act on severity

### classical setup
* focus on symptomps = end-user pains
* monitoring metrics: 4 golde signals
	* latency: time to serve
	* errors: number of failing requests
	* traffic: total number of requests
	* saturation: load on network
* have alerts

### ML setup
* much more complex setup, many more things can go wrong, but none of them are caught by 4 golden signals
* Need more and different monitoring
* Also do symptom based monitoring, prioritize backward from output

* Evaluation metrics
1.  Go for scientific evaluation metrics in production
	* Do we get a true label back, close enough in time?
		* Delayed result
		* Unknown result (e.g. in case of rejecting a customer in fraud detection)
		* Think around this: e.g. holdout set of applications that ALL get accepted, to measure metrics on this
	* Have alerts and dashbords for this
2. Stakeholder fear signal metrics
	* Ask stakeholders for their worst fears, and create metrics from that

* Monitoring metrics/Detection metrocs
1. Monitor response distribution
	* Easier than input monitoring; just few output, no need to care for feature importance
	* Detects both slow and sudden shifts
	* Rule based disrance: e.g. mean shifts by 10%
	* Statistical distance: e.g. Kolmogorov distance, Google's D1 metric
2. Use-case specific, human understandable quality indicator
	* heuristics for "really good" and "really bad" response
3. Monitoring input data
	* monitor input and feature distribution
	* compare difference between training and and serving, or train on features you logged
	* compare serving distribution over time

### Tooling
* if its easy, add to serving code, then put up on dashbord
* if its complicated, logg stuff and run computation every n minutes

# Sign Language To Speech
* Dataset
	* 8000 images from TV broadcasting, 5000 after data cleaning
	* TXT annotations from volunteers
	* Preprocessing: flipping, scaling,
* Model
	* YOLOv5 model: good on classification and bounding box
	* Single shot detector using Resnet50
		* Pretrained using COCO dataset (?)
	* MobileNetv2: good on precision and recall

# Data APIs
* standardisation consortium
* numpy was attempt to consolidate scientific array computations
* but now we have loads of other data science frameworks
* tasks: array API and dataframe API
* people follow numpy, numpy follows this consortium
---
Associations: [[z> MOC PyData 2022]]
Tags: #📥️ (extract ideas from workshop notes)
