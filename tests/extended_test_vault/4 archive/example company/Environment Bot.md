## Summary
Example Company's Environment Bot coordinates temporary environments across
several services and can prepare dependent changes for a shared interface update.

## Example request
The syntax below is illustrative. Service A and Service D are placeholders.

```text
@EnvironmentBot create <environment> service-a/<branch> service-d/<branch>
```

Check which services and source branches are included before requesting a
deployment. Treat environment creation and persistent configuration updates as
separate operations when the tooling distinguishes them.

---
Associations: [[dynamic environment]], [[z> MOC Infrastructure]]
