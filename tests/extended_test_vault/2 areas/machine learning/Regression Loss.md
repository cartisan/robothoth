## Context
- loss function: objective function that is being minimized by an ML algorithm
	- different loss functions for classification and regression
- selection of loss function depends on:
	- size of dataset and time constraints
	- presence of outliers
	- ML algorithm

## Mean Absolute Error
- MAE, also L1 Loss
- Mean absolut distance between target variable and predicted values
	- $MSE = \sum_{i=1}^{n}|y_i - ŷ_i|/n$
- If we use the the directed loss (instead of the absolute) we get the Mean Bias Error (MBE), which is the mean residual and thus not necessarily good even when it's 0
- Since gradient is the same throughout, this loss is harder to learn from especially for neural networks, and can lead to overshooting the minimum
- Also, function is not differentiable at 0, which is a problem for neural networks
- ![[Pasted image 20240903164332.png|350]]
## Mean Square Error
- MSE, also L2 Loss, Quadratic Loss
- Mean squared distances between target variable and predicted values
	- $MSE = \sum_{i=1}^{n}(y_i - ŷ_i)^2/n$
- Outliers have stronger effect on MSE than on MAE, which makes model less robust towards outliers (model tries harder to minimize outlier loss, at the expense of more common samples) but also less likely to produce wildly wrong predictions (that would result in a bigs loss)
- If we take the Root Mean Squared Error (RMSE), we get values on the same scale/unit as our target variable, which is easier to interpret
- ![[Pasted image 20240903164259.png|350]]
## Huber Loss
- Also Smooth Mean Absolute Error
- Returns absolute error for large errors, and quadratic error for small errors, with threshold $\delta$ being a hyperparam:
	- $L_{\delta}=0.5(y_i - ŷ_i)^2~~~~~~~~~~~~~if~~|y_i - ŷ_i| < \delta$
	- $L_{\delta}=\delta|y_i - ŷ_i|-0.5\delta^2~~~~~else$
	- $H_L =\sum_{i=1}^{n}L_{\delta}(i)/n$
- Hyperparam $\delta$ determines what you’re willing to consider as outlier, larger residuals minimized with L1 (which is less sensitive to large outliers), smaller ones "appropriately" with L2.
- Less sensitive to outliers than MSE, differentiable at 0
- ![[Pasted image 20240903164349.png|350]]
## Log-Cosh Loss
- Logarithm of the hyperbolic cosine of the prediction error
	- $L=\sum_{i=1}^{n}log(cosh(ŷ_i-y_i))$
- Is approximately equal to (x² / 2 for small x and to abs(x) – log(2) for large x
- Similar advantages to Huber Loss, but is twice differentiable everywhere (unlike Huber Loss), which can help all algorithms that require the second derivative (Hessian), like e.g [[XGBoost]]
- ![[Pasted image 20240903164405.png|350]]
## Quantile Loss
- A generalization of MAE, that gives different weight $\alpha$ to absolute loss depending on whether its positive or negative
	- $L_{\alpha} = \alpha(y_i - ŷ_i)~~~~~~~~~~~~~~~~~~~~~~~if~~(y_i - ŷ_i) \geq 0$
	- $L_{\alpha} = (\alpha-1)(y_i - ŷ_i)~~~~~~~~~~~~~if~~(y_i - ŷ_i) < 0$
- With $0 < \alpha < 1$ being the quantile, and the Loss equaling MAE for $\alpha=0.5$
- The idea is to choose the quantile value based on whether we want to give more value to positive errors or negative errors
- Can be used to predict [[confidence intervals]], by training models with different quantile values and using their predictions as interval boundaries
- ![[Pasted image 20240903171904.png|350]]

##
Associations: [[default_vault/0 inbox/z> MOC ML]]
Tags: #🌱 
Sources: https://fritz.ai/best-regression-loss-functions/
