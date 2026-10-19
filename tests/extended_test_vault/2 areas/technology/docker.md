## General
- OS system level virtualization (instead of machine level)
	- can put many servers on one machine
	- docker is linux (running inside linux)
- prentend that we have our own machine, that is collocated at our hpst
- docker file is a recipe on how to create a server, and this is built into an immutable artefact

## Dockerfile
	- Contains setup and config of container
	- Base image like ubuntu: `FROM dockerfile/ubunut`
	- Each command is an own layer (like FROM, thats setting up layer 1)
	- When built, this creates an *immutable* docker image artfact (the layers describe the setup from which we always start)
		- the state changed during the session will be gone after restarting the container
- Builidng the file
	- `docker build -t tagname:latest .`  creates the the image with tagname and version 'latest'  in current dir
	- Rebuild after each change of config
- Run imge
	- `docker run tagname:latest` executes the CMD commands
	- `docker ps` shows running docker processes
	- `docker exec -ti CONTAINER ID any command` runs any command in the docker

## Procedure:
```bash
python configure.py
make build
make run
```
