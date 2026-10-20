## Idea
- Can do regression and classification
- train multiple decision trees on subsets of data and features
	- recursively partition data into smaller subsets at each split
	- continue until stopping criterion, like maximal depth or minimal samples per leave, is met 
- prediction is ensemble of individual tree predictions, e.g. majority vote (classification) or mean (regression)

## Advantages
- *Robustness*: less prone to over-fitting due to random splitting
- *Non-linearity*: can capture complex relationships between features and target
- *Interpretability:* provides information about feature importance
## Problems
- *No Extrapolation:* can only predict between min and max (averaged per tree)
- 
##
Associations: [[Extra Trees Classifier]]
Tags: #📥️ 
Sources:
