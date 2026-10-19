## summary

pipenv creates a Pipfile.lock and you use it when you move to a different system.
The Pipfile.lock contains all the dependencies and its versions. When you install Python packages using Pipfile.lock, it will create exactly the same environment as your original system.

## commands
`pipenv shell` activate virtualenv of current project
`pipenv run command`  run an individual command in virtualenv of current project
`pipenv install`  install dependencies from Pipfile.lock
`pipenv --python 3.7 install`  create new environment with python 3.7
`ls ~/.local/share/virtualenvs/` show all installed venvs
`pipenv --rm` remove project virtualenv
`exit` close environment

## problem solving
- when a private package repository can not be fetched
	- `pipenv --clear`
	- check `~/.netrc` file: it might contain old gitlab tokens

- ---
Associations: [[z> MOC Infrastructure]]
Tags: #🌲
