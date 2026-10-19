## Intuition
- Variance measures the degree of dispersion of data around the mean
- The weight of the dispersion is judged by distance from the mean, which makes it different from [[standard deviation]] and MAD
- Values are hard to interpret, because they use a different unit than the feature it is computed over
## Formula
$$\sigma^2 = \frac{\Sigma(x_i-\mu)^2}{N}$$
where:
- $x_i$      are individual values in the data set
- $\mu$       is the mean of all values in the data set
- $N$      is size of the data set​

- The squared distance from the mean is sometimes denoted $\sigma(x,x)$, because the distance from the mean in $x$ is squared
	- this distinguishes variance from [[Covariance - a measure of the joint variability of two variables]], denoted $\sigma(x,y)$, because there the distance from the mean in $x$ is multiplied with the same distance in the other features of a point
	- i.e. variance is the covariance of a measure with itself
- Only explains spread of data along directions parallel to one of the axis of a coordinate system

---
Associations: [[Descriptive Statistics]]
Tags: #🌱
