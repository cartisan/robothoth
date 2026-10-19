## summary
Python-dotenv reads key-value pairs from a .env file and can set them as environment variables. It helps in the development of applications following the 12-factor principles. To configure the development environment, add a .env in the root directory of your project.
Syntax:
```
# Development settings
DOMAIN=example.org
ADMIN_EMAIL=admin@${DOMAIN}
ROOT_URL=${DOMAIN}/app
```
If you use variables in values, ensure they are surrounded with { and }, like ${DOMAIN}, as bare variables such as $DOMAIN are not expanded.

https://pypi.org/project/python-dotenv/

## General

- Container orchestration platform, we are on version 1.20
- Everything in k8s is an API
- Namespaces
    - like a folder, everything should be grouped in proper namespaces
    - match namespace name to repository name
        --\> one 'service' per namespace
- Pods
    - Basic interaction unit is pod, pods contain containers (1:n mapping)
        - in case of many containers you always have a main container and several side cars
    - what components should run, with which env vars and which ressources
- Job is an API spec with attached data that has a pod
    - apply installes a new API called a job into the cluster
    - container for short lived workflow (e.g. meant to run a pod once, then its done)
    - Cubelet (a container) responsible for trasnforming YAML files into pods
    - Has a pod template spec, that has a container spec
- Config map holds key value pairs, for instance for storing env vars
- API Doc: https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.20/

- Probes
  - liveness: can it operate without error
  - readiness: can it serve traffic
  - show up, when we look at output of `kubectl get events`

- Rolling Updates
	- replicas serve pods
	- there is one deployment, that manages replicas
	- once a change is made, the deployment restarts replicas one by one
	- `kubectl get deployment`
	- `kubectl get replicaset`

## Commands

**Remember to use CLI managemet tool:** [[kdash]]

`kubectl get namespaces`
`kubectl apply start_pod.yml` start job with associated pod using yml

`kubectl get pods -A` show all pods
`kubectl get pods -n` show all pods in namespace `n`
`kubectl get pods -o wide` show info in wide/yaml format

`kubectl get pod example-service -o yaml` show info about pod in YAML

`kubectl get events -n example-service --sort-by='{.lastTimestamp}' -w` show events sorted by time stamp

`kubectl get all -n example-service`  shows all things in namespace

`kubectl delete jobs/example-service` delete job
`kubectl delete all --all -n example-development` delete all in namespace

`watch kubectl get all -n example-development` watch stuff appear in namespace

`aws eks --region eu-central-1 update-kubeconfig --name dev`  switch to dev cluster (or prod cluster)

`kubectl get pods -A | grep 'example-development'` get deployments from all namespaces, i.e. Service A, Service B, Service C, ...

---
Associations: [[z> MOC Infrastructure]]
Tags: #🌲
