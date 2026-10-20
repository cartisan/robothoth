Tree based transformers (like `RandomForestRegressor` or `ExtraTreesClassifier`) are sensitive to feature ordering unless `max_features==n_features`. Even just renaming features will have an effect on RF training, in this case.

## Explanation:
- If `max_features < n_features` then the model will randomly sample the appropriate number of features from the set of all features at each split. The random seed will typically be fixed, but then the order of the features will have an effect on which features will be available during splitting.
- C.f. [https://github.com/scikit-learn/scikit-learn/issues/5394](https://github.com/scikit-learn/scikit-learn/issues/5394)
## Lessons
- Typically you'd want your tree based models to take over responsibility for maintaining a stable feature ordering themselves, do decouple them from the internals of the preceding preprocessing or the containing pipeline as much as possible. 

---
Associations: [[scikit-learn]], [[default_vault/0 inbox/z> MOC ML]]
Tags: #🌲
