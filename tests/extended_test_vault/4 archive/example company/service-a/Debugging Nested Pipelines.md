## Debug framework-delegated steps
A nested scikit-learn pipeline delegates execution to the framework, so a
breakpoint in the containing application may not expose each intermediate step.
Inspect the framework's pipeline iteration and place a conditional breakpoint
on the named step being investigated.

For a fictional step called `example-step`, a debugger condition can inspect
the pipeline's `steps` and the current step name. Check the installed library
version for the appropriate location rather than relying on a fixed line number.

Inspect both fitting and transformation when the two execution paths differ.

---
Associations: [[scikit-learn]]
