## Basics
* time series: array of data points aligned with an array of time stamps
* period: frequency of time stamp entries, e.g. every day
* connection to ML: 
	* ML detects patterns; time series encode patterns over time --> predict future
	* steps
		* feature extraction
		* model fitting
		* validation
## Feature Engineering
- aggregation features like: min, max, mean
	- use `.rolling(window_size).aggregate([func1, func2, ...])`
	- the functions in aggregate should take only one parameter: the data in the window
		- to use a configurable functions, like e.g. `np.percentile`, use `functools.partial`: 
		   ```python
		   percentile20 = functools.partial(np.percentile, 20)
		   data.rolling(20).aggergate(percentile20)
		   ```
	- python package `tsfresh` computes many features for each window
- date based features
	- weekday, month, year, holiday
- audio:
	- envelope (total amount of energy present at each time point)
		- rectify: i.e. make all values positive via `np.abs`
		- smooth time series by using [[pandas#Windowing|rolling windowed mean ]]
	- spectorgram
		- squared windowed Fast Fourier Transform
- predicting future time points from previous time points
	- how smooth is data, aka: what's the amount of auto-correlation?
	- ```python
	  data = pd.Series(...)
	  shifts = [0,1,2,3,...]
	  shift_dict = {f"lag{i}": data.shift(i) for i in shifts}
	  features = pd.DataFrame(shift_dict)
	  ```
## Data cleaning
- handle missing data
	- interpolate: e.g. using pandas `df.interpolate` method
	- impute:
- normalize data
	- normalize each value by percent change between last value and mean of previous values
		- using a rolling window
		- in each window: $x_{norm} = (X_{-1} - \overline{X_{:-1}}) / \overline{X_{:-1}}$ with X being the values in the window
		- de-trends the data 
- in case of non-stationary time series data
	- to test whether a time series is stationary, run xval and compute statistics for each fold
	- apart fro de-trending, we can also limit the size of the training data to only recent samples
- handle faulty data
	- constant data:
		- implies device/measuring error -- treat like outliers?
	- outliers:
		- any datapoint that is more than 3 std away from mean
		- detect after normalizing the data
## Metrics
- accuracy/precision for classification
- [[R² -Coefficient of Determination|coefficient of determination]] for regression
- crossvalidating time series data
	- ```python
	  from sklearn.model_selection import TimeSeriesSplit
	  cv = TimeSeriesSplit(n_splits=10)
	  for train, test in cv.split(X,y):
	      model.fit(X[train], y[train])
	      pred_y = model.predict(X[test])
	      print(r2_score(y[test], pred_y))
	  ```
- often, it makes sense to relate forecast accuracy to business value and use that as a metric
	- e.g. when forcasting demand of a product, how does our forecast error translate into additional/missing batches ordered?
- aggregation matters in these metrics
	- aggregating accuracy/error over a week will give different results than aggregating over 7 days and then taking the mean
	- the right aggregation depends on business decisions - e.g. do we plan weekly, or daily?
##
Associations:
Tags: #📥️ 
Sources: 
* https://app.datacamp.com/learn/courses/machine-learning-for-time-series-data-in-python
* https://www.relexsolutions.com/resources/measuring-forecast-accuracy/  (relating metrics and bussiness decisions)
