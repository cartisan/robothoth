- If deploying/releasing code is risky, it is a priority to fix that since its a self-reinforcing state
	- when deploys are risky, people deploy less and change accumulates, which makes deploys even more risky
	- risky deployments increase the [[Friction is the difference between the model and the world |friction]] on EACH AND EVERY release
- When fixing this, be aware that this is a project (that will take time)
	- assign an owner
	- track metrics that will show your progress
	- practice [[Post Mortem - Incident Report|post mortems]]
- Fixing this is about making your CI/CD pipeline resilient to errors
	- this requires identifying common failure cases and automatically recovering from them

## Prioritize Deployment Software
- deploy software is the most important software you have
	- it ships new code, and helps recover from problems
- Recruit an owner for it and allocate time for development and maintenance, testing and tracking
- signal high organizational value for it
	- e.g. put the best engineer on it
- Track how often deploys fail and why, have a manager pay attention to this metric

## Ship early and often
- smaller change sets are easier to deploy, debug and track when something goes wrong
- don't trust code until it has been on prod for a while
	- it needs to see a lot of different load before we know it behaves
	- don't rely on never failing: expect failure  and build guard rails that allow debugging and fixing them quickly
	- Look at what you have deployed and own it in case of failure
- "It takes time to bake a new release"
- Tools to enable this:
	- feature flags to switch code paths on or off
	- build canaries that allow releasing change sets to bigger and bigger subsets of users

##
Associations: [[Software Engineering]]
Tags: #🌲 
Sources: https://charity.wtf/2018/08/19/shipping-software-should-not-be-scary/
