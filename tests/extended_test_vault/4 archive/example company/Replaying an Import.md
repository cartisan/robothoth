## Replay an import
Example Company uses Service E for scheduled data imports. Before replaying an
import, confirm the target environment and prerequisite configuration records.

An illustrative interface accepts an input source, environment, start date,
optional skip count, and file type. These are placeholders, not production IDs.

```bash
<import-cli> replay --source <input-source> --environment <environment> --start-date <date>
```

The replay places input files on the selected queue; a scheduled worker consumes
them. A replay does not recreate unrelated application configuration. Check that
configuration separately and verify the import outcome through the work logs.

---
Associations: [[z> MOC Infrastructure]]
