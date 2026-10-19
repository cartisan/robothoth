## Keep a compatible runtime available
An incompatible Service A deployment can fail to load existing serialized models
while replacement models are being trained. Temporarily retain the old tagged
serving jobs until the replacements are ready.

## Legacy jobs
1. Copy the old Argo workflow template and its cron job to `*-legacy` names.
2. Keep the image reference `<old-image-tag>` so the old artifacts use their
   compatible runtime.
3. Remove generated or controller-managed fields, including status, UID,
   resourceVersion, generation, and Helm-managed metadata.
4. Point the legacy cron job's `templateRef` at the legacy workflow template.
5. Verify that the replacement artifacts are ready before retiring the legacy jobs.

Service names, version tags, and deployment identifiers are placeholders. This
runbook describes deployment coordination, not a model's algorithm or inputs.

---
Associations: [[Model Compatibility and Release Versions]], [[Model Artifact Runtime Compatibility]], [[kdash]]
