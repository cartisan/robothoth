## Artifact and runtime compatibility
A serialized model artifact may depend on the code and library versions that
created it. A deployment can become incompatible with an existing artifact before
its replacement is ready.

## Possible approaches
- Record the runtime associated with an artifact and make it available for loading.
- Keep the older serving job running until the replacement artifact is ready.
- Evaluate a version-aware state migration when the compatibility contract permits it.

These are design options, not an instruction to apply all approaches together.
The release process must still account for bug fixes and the time needed to
prepare replacement artifacts.

---
Associations: [[Model Compatibility and Release Versions]], [[Running an Older Service Version During Retraining]]
