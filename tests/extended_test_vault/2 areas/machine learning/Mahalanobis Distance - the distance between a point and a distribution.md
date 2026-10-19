## Intuition

- Distance metric that takes into account the variance and covariance of the distribution
	- [[Variance - a measure of data spread that gives more weight to outliers]]
	- [[Covariance - a measure of the joint variability of two variables]]
	- An Euclidean distance would just measure the distance to the mean of the distribution along both dimensions
		- Disregards that the spread along one dimension might be much higher than along the other e.g. a distance of 3 along $x$ might mean a value is inside the distribution, but 3 along $y$ might mean it is outside the distribution if $\sigma^2_x > \sigma^2_y$

- Lets the (multivariate) data decide how to construct a coordinate system in which distances can be measured![[Pasted image 20230802151408.png|center|300]]
	1. Define coordinate axis: one in the direction along which data has most variance (the spine), other perpendicular to that, with $(0, 0)$ at distribution centroid.![[Pasted image 20230802151712.png|center|300]]
	2. Define one unit along each axis as the standard deviation along this axis![[Pasted image 20230802151729.png|center|300]]Scale the space, such that one unit horizontally equals one unit vertically![[Pasted image 20230802151747.png|center|300]]
- In the Mahalanobis space
	- the axes are placed such that distribution is decorrelated (turning covariance into variance), which makes it related to [[PCA]]
	- unit vectors along the new axes are the eigenvectors of the covariance matrix (making the variance of the transformed distribution $1$)

## Formula
$$D² = (x-m)^T \cdot C^{-1} \cdot  (x-m)$$
where: 
 - $D^2$        is the square of the Mahalanobis distance. 
 - $x$          is the vector of the observation (row), 
 - $m$         is the vector of mean values of the independent variables (mean of the columns), 
 - $C^{-1}$     is the inverse [[Covariance - a measure of the joint variability of two variables#Covariance Matrix|covariance matrix]] of the independent variables

Explanation:
- $x-m$ is the distance of a sample from the mean point
- $(x-m)^T \cdot C^{-1}$ divides this distance by the covariance, in essence computing a multivariate $z$ value (think:  $z=\frac{x-\mu}{\sigma}$ )



---
Associations: [[default_vault/0 inbox/z> MOC ML]], [[Features]]
Tags: #🌱
