## Philosophy
- Never import things from testing but only from production code
	- Everything you would like to import from testing should be a conftest fixture and passed via method arguments
	- annotations can declare scopes in which things created in conftest are cached between runs (e.g. sessions or packages)

---
Associations: [[python]]
Tags: #🌱  #🔭
