## Idea
- Uses *regularization* techniques to avoid over-fitting
- Combines *Lasso Regression Penalty* and *Ridge Regression Penalty*
	- Lasso: good when many useless parameters are present, eliminates them
	- Ridge: good when many useful parameters are present, it shrinks parameters but does not remove them
	- Each of the two get their own weight $\lambda$ , and cross validation is used to find best combination of weights
- Starts with optimizing sum of squared residuals, and combines Lasso with Ridge regularization combining the best of both worlds so we don't have to chose
- Good at dealing with correlated parameters, since it removes correlated parameters together or keeps them all at once

### Ridge Regression Penalty
- MSE can over-fit with few training data, so we want to introduce a bias in the error term to avoid that
- New minimization function: MSE + $\lambda \times slope^2$ (weight of single parameter)
- with a smaller slope, differences in a parameter result in less changes in predicted value, so Ridge Regression introduces a bias towards less sensitivity to the parameter
- If we have multiple parameters, then we have a weight $\lambda_i$ and squared penalty coefficient for each parameter

### Lasso Regression Penalty
- like ridge regression, but regression penalty is MSE + $\lambda \times |slope|$
- for this reason, Lasso can shrink the slope to 0
## 
Associations: [[default_vault/0 inbox/z> MOC ML]]
Tags: #🌱
