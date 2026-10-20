- When you unpickle an object, the `___init__` method of its class does not get executed
- That makes sense, as executing the initialize method would overwrite the unpickled state
- That means that if you change some code in the init of an  [[scikit-learn]] type pipeline, this won't be picked up in running code until the model is retrained
- **This can result in broken code seemingly running fine on staging, if it is part of the init**

---
Associations: [[python]]
Tags: #🌲
