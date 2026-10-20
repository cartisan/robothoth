## Idea
When you get an unknown dataset, you need to get an idea of it in order to be able to implement proper data preprocessing and decide on appropriate model.

## Process
1. Get overview of tables
	- what tables are there with what shapes
	- Look at each table: columns, and example rows
	- connections?
2. Get overview of target
	- target distribution: are there multiple modes, can they be explained by groups?
	- Boxplots for outlier detection, perhaps per group
	- is there a time feature? Plot targets over time, perhaps per group
	- How are datapoints with targets distributed between groups?
3. Get overview of features
	- number of datapoints per feature (do some points miss certain features?)
	- box plot of features (per group) to find outliers and variance for features (low variance means useless feature)
	- Distribution of feature values: skewdness and modality
	- correlation of features with target and with themselves (to find important but also co-linear features)

---
Associations: [[default_vault/0 inbox/z> MOC ML]], [[Markus Norden (Senior ML Engineer)]], [[Yasser Sheikh (Senior ML Engineer)]], [[Andre Ferreira (Senior ML Engineer)]]
Tags: #📥️ (TODO: needs some theoretical groundwork, not just observations from interviews)
