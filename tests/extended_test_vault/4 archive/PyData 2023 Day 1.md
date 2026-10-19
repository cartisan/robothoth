## Interesting talks
- 10:50: From notenbook to pipeline
- 11:40: HPO for the impatient
- 15:10: Staying alert - continuous testing for ML monitoring
- 15:45 -- break or Polars
- 16:20: Observability into ML

## LineaPy: from notebook to pipeline in no time
LineaPy tracks how your variables come to their values during notebook execution, by watching memory. It allows you to create pipelines that can be dockerized, right form the notebook.

### Experimenting
- Use LineaPy in your notebook, so you know how to linearize your notebook
- Watches memory to know how variables came to the values they are having
- Also watches imports, so if you use numpy for anything related to a vriable, it will know that as a dependencie
```notebook
!pip install lineapy
%load_ext lineapy
import lineapy

a=1, b=2
c=a+b
artifact = lineapy.save(c, "c")
print(artifact.get:code())
>> a = 1, b = 2
>> c = a + b

import numpy
my_param = 3
d = np.dot(np.array[c], np.array[my_param])
d
>> 9
```

### Export to pipeline
```notebook
dir = lineapy.to_pipeline(
  ["c", "d"],
  framework="DVC",
  input_params=["my_param"],
  name="my_pipeline",
  dir="./"
)
```
- creates a `my_pipeline_module.py`
- analyses code and dependencies to create pipeline steps in a framework of your choice, e.g. DVC
- creates a dockerfile to run the pipeline
- ==Never review a notebook again!==

## HPO for the impatient
- ==Syne Tune: open source python HPO library==

### Full training
- Grid Search and Random Search: do not adapt over time
- BaysianOptimization
	- uses surrogate (baysian) model to estimate response function that takes hyperparams and gives model performance
- All three are expensive

### Multi-Fidelity Optimization
- train on less data or fewer epochs (Assumption: Lower fidelity is representative of higher fidelity)
- gives approximation of performance
- Successive Halving (SH)
	- Samples Random Configurations, train them for lowest compute budget
	- Kick out 50% worst models (usually, people throw out more like 2/3)
	- Double Budget, continue training remaining models, repeat algorithm
	- What is a good minimal budget?
		- Try different brackets --> new hyper-hyper parameter
	- When do you stop? (Maximal budget?)
- PASHA (extends SH)
	- Chooses maximal budget for you (i.e. when do you stop?)
	- Checks when ranking of trained models becomes stable, then stops
		- due to noise that happens rarely, so use some soft-ranking
- RUSH (extends SH)
	- Assume we know a set of good default settings
	- Like SH: but also drop configurations that are doing worse than good default settings
	- In case of good transfer: drops bad candidates really fast
	- In case of bad halving, like vanilla SH

## Python Keynote (Susan Shu Chang)
- RenPy: visual novel game engine
- Bullshit

## Common Issues w/ Time Series Data
- General pipeline:
	1. Pull in data
	2. Analyse and find problems with data
	3. Solve issues
	4. Take step back and think about actual problem
- Tips:
	- Have a time stamp for every piece of collected data or every step of transformation
	- when you have a growing variance () in your data, sometimes it's because you have "growth" as a confounding variable -- you can normalize by things like number of customers
	- imputation: used exponentially smoothed values

## How to build observability
- Monitoring: Catching problematic behavior
- Observability: Understanding what went wrong
- 3 production challenges

### Production Challenge: Data Quality
- Data errors cause unexpected behaviors in model
- Addressed in Feature Store
	- Defines validation via data quality rules
	- Actual users/domain experts define the rules
	- Have dashbords show how often validation breaks, which percentage of data is being blocked

### Data Drift
- distribution of data changed from training time to inference time
- Addressed in ML Pipelines, by injecting distribution monitoring steps
	- After each step of pipeline, make a snapshot of the outgoing data
	- Compute statistical profiles of this data using [WhyLogs](https://github.com/whylabs/whylogs)

### Model Performance
- When your model isn't working as expected
- Addressed in Online Endpoint
	- Model Outages: model unavailability
	- Anomalous Inputs: ==handle imperfect data, but at the same time be able to keep track of it==
	- Low Latency: response latency
	- Rolling Statistics: during inference, keep rack of changes in distribution over time

---
Associations: [[z> MOC PyData 2023]]
Tags: #🌲
