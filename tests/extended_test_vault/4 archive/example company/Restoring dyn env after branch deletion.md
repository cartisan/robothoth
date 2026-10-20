## Problem
Deleting a branch can schedule a temporary environment's secret for deletion.
A later environment-creation job may fail while that scheduled secret still
occupies its name.

## Recovery
Confirm the temporary environment and secret identifier in the authorized secret
store. Follow the environment's recovery procedure to resolve the scheduled
deletion before recreating it. Do not infer a production identifier from an example.

Example identifiers: `<environment>`, `<secret-id>`, and `<service-repository>`.

---
Associations: [[dynamic environment]]
