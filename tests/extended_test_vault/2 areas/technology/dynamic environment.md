## Create a temporary environment
Example Company uses a temporary environment for isolated development work.
The service and environment names are illustrative placeholders.

```bash
<environment-cli> create example --services service-a
```

Set `ENVIRONMENT_SLUG` in the local configuration to the selected environment.

## Access configuration
Retrieve connection settings from the authorized secret store. Do not copy
credential values into notes or repository fixtures.

```bash
aws secretsmanager get-secret-value --secret-id <secret-id>
```

Illustrative output:
```json
{"SecretString": "<secret-value>"}
```

## CI/CD
Use the documented environment marker for the deployment pipeline. Confirm
that the selected environment is isolated before making schema changes.

---
Associations: [[z> MOC Infrastructure]], [[Environment Bot]]
