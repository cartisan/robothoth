- A transformation that create features of higher order, defined by a degree
	- A polynomial feature trafo of 2nd degree turns a sample (a, b) into (1, a, b, ab, a², b²)
	- If we only care about interaction terms, then we get (1, a, b, ab)
	- type of feature engineering
- Usually low degree such as 2 or 3, otherwise polynomial curve becomes overly flexible which can lead to overfitting
- Works well with linear models, as it allows them to capture non-linear patterns
	- a linear model with polynomial features is called *polynomial regression*
- Available as [trafo in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
##
Associations: [[default_vault/0 inbox/z> MOC ML]], [[Features]], [[scikit-learn]]
Tags: #🌲 
Sources: 
- https://scikit-learn.org/stable/modules/preprocessing.html#polynomial-features
- https://machinelearningmastery.com/polynomial-features-transforms-for-machine-learning/
