## Interesting talks
- 10:30: State of Production ML


## State of Production ML
- Slides: bit.ly/statemlops
- [Has PyCon Keynote on ML Monitoring on Youtube](https://www.youtube.com/watch?v=QcevzK9ZuDg)
- Challenges: Specialized hardware, complex data flow, reproducibility
- Anatomy of ProductionML system:
	- Components: Training Data, Artifact Store, Inference Data
	- Process: Experimentation, CI/Deployment, Model serving, Monitoring
	- Metadata on whole process
	- --> This is the *canonical ML stack*, and we can use different tools for each part of it
- Monitoring:
	- Performance Metrics: 4 golden signals
	- Statistical Performance: performance metrics like accuracy
	- Drill Down on Metrics: be able to compare statistics
	- Explainability: why did model predict something?
- Observability:
	- Alerting
	- Automated SLOs/SLIs: measuring performance promises
	- Progressive Rollouts: don't deploy to everyone at the same time
	- Drifts and Outlier Detection: non manual approach
- DataMesh:
	- DataLakes are not silver bullets
	- Have Data Lake with all data, and then have hubs of specialization
- Meta Data:
	- Ability for you to know, what the digital assets are that you have running at the moment
	- How do you manage meta data on all of the previous steps?
- Security Risk Areas:
	- Publish: [MLSecOps Top 10](https://ethical.institute/security.html)  <-- top 10 ML security risks right now

## Chatbot for internal data
- Heterogenous data
	- longer documents: have more structure through headers
	- slack messages: spam, events, relevant information
- Reading data via APIs
- Creating additional meta data:
	- spam vs. no spam classification
	- Use SetFit algorithm
- Save/Retrieve
	- Save documents in vector db
	- Retriever
		- use vector-based retrievers via embedding vectors
	- Reader
		- ...
- Used technologies: 
	- Haystack
		- Search for Django
	- [Weaviate](https://weaviate.io/) 
		- Vector DB
		- Stores objects along with ML generated vectors so that retrieval based on vectors is fast

## 5 Things about FastAPI I wish I new before
- Pydantic
	- Use `Extra.forbid` in pydantic, as default is `true` and it will allow saving previously undefined parameters
	- Inheritance: validators are inherited, too, which might lead to unexpected problems

## Ensemble programming
- Principle: Treat each other with Kindness, Consideration, Respect
- Do a retro after the session
- The bigger a mob gets, the faster you want to rotate
	- If you have 6 people you could change every 2mins, versus 15mins for 2 people
- "runaway typist" ... typist who doesn't do what speaker says
- ==Rotation order: person who is typing switches to person who is talking==

## Bumps in the road: data viz
- dynamic vs. static?
- has CI/CD for visualizations!
	- generates a pdf using a github workflow
- Principles
	- less is more: reduce as much visual distractors as possible
	- 

---
Associations: [[z> MOC PyData 2023]]
Tags: #🌲
