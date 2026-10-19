## Intuition
- Is a measure of correlation between the jointly distributed variables $x$ and $y$
	- is negative in cases of inverse correlation, i.e. sign shows tendency of linear relationship
	- is symmetrical, i.e. $\sigma(x,y) = \sigma(y,x)$
## Formula
$$\sigma(x,y) = \frac{\Sigma((x_i-\mu_x)(y_i-\mu_y))}{N}$$
where:
- $x_i, y_i$      are corresponding, individual values in a multi variate data set
- $\mu$            is the mean of all values according along one feature
- $N$           is size of the data set


- Related to [[Variance - a measure of data spread that gives more weight to outliers]]
	- variance is the covariance of a variable with itself
	- variance only measures spread of data in directions parallel to one of the axis of a coordinate system (shape), while covariance explains whether these variations are coordinated (orientation)
### Covariance Matrix
- A covariance matrix over 2 dimensional data is defined as:$$\Sigma = \begin{bmatrix}\sigma(x,x) & \sigma(x,y)\\ \sigma(y,x) & \sigma(y,y)\end{bmatrix}$$
	- Examples: ![[Pasted image 20230803112331.png| center | 300]]


---
Associations: [[Descriptive Statistics]]
Tags: #🌱
