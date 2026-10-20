## Infrastructure as code
Terraform describes infrastructure resources and uses provider APIs to manage
them. State records the resources associated with a configuration. Concurrent
or out-of-date changes can make a plan differ from the intended infrastructure.

## Example Company workflow
- Keep infrastructure definitions in `<infrastructure-repository>`.
- Separate development and production state.
- Use the agreed shared state backend and locking for collaborative work.
- Review `terraform plan` before applying a change.
- Work from an up-to-date branch and confirm the selected state and environment.

## Terms
- A resource describes a managed object.
- A module groups related configuration.
- `plan` previews proposed changes; `apply` executes an approved plan.

---
Associations: [[z> MOC Infrastructure]]
