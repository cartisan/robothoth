## General outlier-detection methods

### Source: 
https://towardsdatascience.com/5-outlier-detection-methods-that-every-data-enthusiast-must-know-f917bf439210

- [Local Outlier Factor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
	- compare local density of focus point with local density of neighbors
	  ```python
	  from sklearn.neighbors import LocalOutlierFactor
	  import numpy as np
	  lof = LocalOutlierFactor(n_neighbors=1, metric='manhattan')
	  data = np.array([3,3,1,1,1]).reshape(-1,1)
	  lof.ft_predict(data)
	  >>> array([1, 1, 1, 1, 1])
	  data = np.array([3,3,1]).reshape(-1,1)
	  lof.ft_predict(data)
	  >>> array([1, 1, -1])
	  ```
- IsolationForrest
	- a tree-based algorithm that tries to keep splitting the data at random thresholds and features till every point gets isolated
	  ```python
	  from sklearn.ensemble import IsolationForest
	  import numpy as np
	  iforest = IsolationForest(n_estimators=5)
	  data = np.array([3,3,1,1,1]).reshape(-1,1)
	  iforest.ft_predict(data)
	  >>> array([1, 1, 1, 1, 1])
	  data = np.array([3,3,1]).reshape(-1,1)
	  iforest.ft_predict(data)
	  >>> array([1, 1, -1])
	  iforest.fit_predict(np.array([3,3,1,1,1],).reshape(-1,1))
	  >>> array([-1, -1, -1, -1, -1])
	  ```

### Source:
https://towardsdev.com/outlier-detection-using-iqr-method-and-box-plot-in-python-82e1e15232bd

- IQR Method
	- outliers are located outside of 1.5 IQR  around the Q1/Q3 band
	 ```python
	 import numpy as np
	 data = np.array([3,3,1,7, 1,1])
	 Q1,Q3 = np.percentile(data , [25,75])
	 IQR = Q3 - Q1  
	 ul = Q3+1.5*IQR  
	 ll = Q1-1.5*IQR
	 outliers = data[(data > ul) | (data < ll)]
	 ```
