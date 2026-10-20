## Basics
- data should be: (samples x features)
	- first dimension is samples (rows), second is features (columns)
- all models offer:
	- `.fit(X,y)`
		- y should be of shape (samples, 1)
	- often:  `.coef_` with learned coefficients
	- `.predict(X)`

## Evaluations
- `sklearn.metrics`:
	- `accuracy_scorer(labels, predictions)` - % of correct predictions
	- `r2_score(y_true, y_pred)` - coeff. of determination (error normalized by variance)
- `sklearn.model_selection`:
	- `cross_val_score(model, X, y, cv=3, scoring=r2_score)` - runs xval with `cv` folds of the data, using r2 to evalaute each split
		- `cv` can also take a custom cross validation object like `TimeSeriesSplit`
		- 

##
Associations: [[pandas]], [[numpy]], [[python]], [[ML for Time Series]]
Tags: #📥️ 
Sources:
