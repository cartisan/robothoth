## Incompatible changes
For Service A, a change incompatible with saved model artifacts requires a
versioned implementation and a new active model version. This is a compatibility
decision, not an automatic consequence of every code edit.

### Release procedure
1. Create `implementation_v<version>.py` and its corresponding tests using
   `<version-migration-script>`.
2. Increase the illustrative `ACTIVE_MODEL_VERSION` setting.
3. Let the deployment's `<schema-setup-step>` activate the corresponding schemas.
4. Prepare replacement model artifacts before retiring the compatible old runtime.

~~Increase `LEGACY_CONFIG_VERSION` to trigger the replacement.~~ **Deprecated:**
the separate configuration counter is no longer used; configuration changes are
detected during deployment.

## Compatible changes
Changes inside methods or configuration can be compatible with saved artifacts.
When compatibility is preserved, a new versioned implementation is unnecessary.
An initialization change still needs testing through fresh object creation:
unpickling an existing object does not execute its `__init__` method.

---
Associations: [[Unpickling sklearn models does not execute their initialization method]], [[Model Artifact Runtime Compatibility]]
