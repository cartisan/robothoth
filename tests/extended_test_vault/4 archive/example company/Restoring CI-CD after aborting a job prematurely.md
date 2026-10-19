## Problem
An interrupted Helm deployment can leave a release marked `pending-upgrade`,
blocking another deployment with an operation-in-progress error.

## Investigation and recovery
Inspect the release history and status in `<namespace>`. An aborted deployment
may not have completed its atomic rollback. Recover the release through the
team's approved Helm recovery procedure; verify the active release afterward.

Example Company keeps enough release history to support rollback and records
which operation was interrupted. Namespace and release names are placeholders.

---
Associations: [[kubernetes]], [[kdash]]
