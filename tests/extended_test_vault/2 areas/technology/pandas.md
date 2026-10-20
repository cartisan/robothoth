## Methods
* `iterrows` is slow, don't do it if you can avoid
* `iloc` stands for *integer* location, not index location (!!!)
* `join`: concatenate series to df along axis 1 (add as column)
* `DataFrame.from_records` translates a list of dicts into df:
```python
>> data = [{'mae': 2.3, 'mse': 8.3},
>>         {'mae': 2.2, 'mse': 8.1}]
>> DataFrame.from_records(data)
     mae   mse
 1   2.3   8.3
 2   2.2   8.1
```
- `query` [[Pandas query() method for more readable data selection|allows more readable data selection using a query language]]
- `reindex` can be used to change the index, but also to select a number of columns from a df and add those columns that do not exist pre-filled with NaN (more useful than just `df[["a", "b"]]` which will fail if  column `b` does not exist):
```python
>> df
           http_status  response_time
Firefox            200           0.04
Chrome             200           0.02

>> df.reindex(columns=['http_status', 'user_agent'])
           http_status  user_agent
Firefox            200         NaN
Chrome             200         NaN
```
- `df[[column_name]]` makes a df with one column from df (while `df[column_name]` only returns a series)
- `.interpolate(linear)` : fills NaNs with (linearly) interpolated values

## Tasks
- to change only a subset of columns in a df, select them as part of the assignment operation
```python
mpa_columns = [col for col in data_with_targets.columns if "press" in col]  
df[mpa_columns] = df[mpa_columns] * 145
```

- exclude one column from selection: `df.loc[:, df.columns!='col_name']`

## Settings
+ reading data using PyArrow backend
	+ faster columnar data format
	```python
	pd.read_csv(my_file, dytpe_backend='pyarrow')
	df = pd.DataFrame([[1, 2], [3, 4]], dtype="uint64[pyarrow]")
	```
+ Copy-on-Write
	+ A new lazy copy mechanism that defers the copy until the object in question is modified was added to the methods listed in [Copy-on-Write optimizations](https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html#copy-on-write-optimizations). These methods return views when Copy-on-Write is enabled, which provides a significant performance improvement compared to the regular execution
	+ https://pandas.pydata.org/docs/user_guide/copy_on_write.html
	+ can be enabled through one of
```python
# globally
pd.set_option("mode.copy_on_write", True)
pd.options.mode.copy_on_write = True

# localy
with pd.option_context("mode.copy_on_write", True):
    ...
```

## Time Series with pandas
- use datetime column as index when representing time series data
### Timestamps
- creatable from strings
- cam access parts of the date
- usable with periods (convert there and back)
- you can use arithemtics for periods, like adding a month
```python
>>> time = pd.Timestamp("2025-01-31")
>>> time.year
2025
>>> period = pd.Period('2025-01', 'M') # gives month end
>>> period.asfreq("D")
Period('2025-01', 'D')
>>> period.to_timestamp()
Timestamp('2025-01-01 00:00:00')
```
### daterange
- a sequence of dates
- returns an DateTimeIIndex
- convertable to period index
- usable as index in data frames
```python
>>> index = pd.date_range(start="2025-01-01", periods=12, freq="M")
DatetimeIndex(['2025-01-31', '2025-02-28', '2025-03-31', '2025-04-30' ...
>>> index.to_period()
PeriodIndex(['2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06', ...
>>> data = np.random.random(size=(12,2))
>>> pd.DataFrame(data=data, index=index).info
DatetimeIndex: 12 entries, 2025-01-31 to 2025-12-31
Freq: M
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   0       12 non-null     float64
 1   1       12 non-null     float64
dtypes: float64(2)
memory usage: 288.0 bytes
```
###  Loading data and transforming it into usable timeseries
- convert string index into datetime index
- access periods using `.loc`
- use `.asfreq()` for up/down sampling
```python
>>> google = pd.read_csv("google.csv", parse_dates["date"], index_col="date")
>>> google["price"].plot(title="google stock price")
>>> plt.show()
# shows plot with reasonable x axis labels
>>> google.loc["2025"]
               price
2025-01-31  0.893670
2025-02-28  0.400881
>>> google.asfreq("D").head()   # upsampling
               price
2025-01-31  0.893670
2025-02-01       NaN
      ...
2025-02-28  0.400881
>>> 
```
### Manipulating time series
- `shift(2)`: shifts data 2 periods into future, while -2 would shift it into the past
- `.div(colname2)`: divides one column by other column (`.sub`, `.mul`)
- `diff()`: difference in value between two adjacent periods
- `pct_change()`: percent change for two adjacent periods
```python
>>> google["shifted"]= google["price"].shift(1)
               price   shifted
2025-01-31  0.893670       NaN
2025-02-28  0.400881  0.893670
>>>google["change"] = google["price"].div(google["shifted"])
```
### Resampling
- use `.asfreq()` for up/down sampling
	- `method` use `ffil` or `bfill` for forward/backward filling
	- `fill_value` to set a certain value
- use `.reindex` to assign a new index to a df, pandas will use same filling logic as with `asfreq`
- `.resample`
	- similar to `.groupby` -- group data and apply methods to each group
		- `as_freq`  just use new offset date
		- `.ffill`/ `.bfill` - upsampling
		- `.interpolate` upsamling -linearly interpolate between existing data
		- `.mean`/ `.sum` downsampling
		- `.agg`  - list of several ggregation functions, each will create new column
	- offset argument: which date in period to use as index (e.g `"MS"`for month start)
	- creates Resampled object
```python
# Upsampling
>>> dates = pd.date_range(start=2025, periods=4, freq="Q")
2025-03-31   1
2025-06-30   2
...
>>> dates.as_freq("M")
2025-03-31   Nan
2025-04-30   Nan
2025-04-31   1
...
>>> dates.resample("MS")  # month start
<pandas.core.resample.DatetimeIndexResampler object at 0x1039beb90>
>>> df.resample("MS").bfill()
1970-03-01  1
1970-04-01  2
...
>>> 
>>> 
>>> 
```
## Windowing
- rolling
	- `.rolling`: move a window of a certain size, apply aggregation on each
		- window: size of window, can be integer, but also string: `"30D"`
		- min_periods: minimum required values for first window
- expanding
	- `.expanding`: add each data point receives a window up until this datapoint (incl)

---
Associations: [[python]]
Tags: #🌱  #🔭
