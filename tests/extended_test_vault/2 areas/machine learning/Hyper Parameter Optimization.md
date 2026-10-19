- Have a model M, its hyperparameters P, as well as a validation set D such that the performance of M using P over D can be given by f(P)
- Task is finding those P that maximize f(P)
- Frameworks: [SyneTune](https://github.com/awslabs/syne-tune), [AutoML](https://www.automl.org/automl/hpo-overview/)

## Black Box Optimization
- Random Search or Grid Search
	- do not search intelligently, so take a lot of time
- Baysian Optimization
	- learns a surrogate (baysian) model to estimate response function f(P)
	- allows exploring more promising P

## Multi Fidelity Optimization
- Since hpo is costly, train on less data or fewer epochs when exploring parameter space
	- Assumption: Lower fidelity is representative of higher fidelity
	- Gives approximation of final performance

- **Successive Halving (SH)**
	- Samples Random Configurations, train them using your minimal compute budget
	- Kick out 50% worst performing models (usually, people throw out more like 2/3)
	- Double last compute budget, continue training remaining models, repeat algorithm
	- What is a good choice for minimal budget?
		- Try different brackets --> hyper-hyper parameter
	- When do you stop? (= maximal budget?)
- **PASHA (extends SH)**
	- Chooses maximal budget for you (i.e. when do you stop?)
	- Checks when ranking of trained models becomes stable, then stops
		- due to noise that happens rarely, so use some soft-ranking
- **RUSH (extends SH)**
	- Assume we know a set of good default settings
	- Like SH: but also drop configurations that are doing worse than your good default settings
	- In case of good transfer: drops bad candidates really fast
	- In case of bad transfer, behaves like vanilla SH

- Lends itself more naturally to neural network architectures where we have to train over same data for several epochs

---
Associations: [[default_vault/0 inbox/z> MOC ML]]
Tags: #🌱
